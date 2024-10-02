# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
import torch.nn.functional as F
import time
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.accelerator import get_accelerator

from dschat.utils.utils import print_rank_0, unwrap_model_for_generation

def print_all_ranks(tag, value, rank):
    world_size = torch.distributed.get_world_size()
    all_tensor = torch.zeros(world_size, dtype=torch.float32).to(
        get_accelerator().current_device_name())
    all_tensor[rank] = value
    torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)
    print_rank_0(f'{tag} {all_tensor}', rank)


def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = hasattr(
                param,
                'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            with deepspeed.zero.GatheredParameters(param,
                                                   enabled=should_gather):
                total += float(param.float().norm())

    return total


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class DeepSpeedPPOTrainer():

    def __init__(self, rlhf_engine, args):
        self.rlhf_engine = rlhf_engine
        self.actor_model = self.rlhf_engine.actor
        self.critic_model = self.rlhf_engine.critic
        self.ref_model = self.rlhf_engine.ref
        self.reward_model = self.rlhf_engine.reward
        self.tokenizer = self.rlhf_engine.tokenizer
        self.args = args
        self.max_answer_seq_len = args.max_answer_seq_len
        self.end_of_conversation_token_id = self.tokenizer(
            args.end_of_conversation_token)['input_ids'][-1]
        self.z3_enabled = args.actor_zero_stage == 3
        self.compute_fp32_loss = self.args.compute_fp32_loss

        # In case the generated experience is not valid (too short), we use the last valid
        # generated experience. Alternatively, we can skip the step (on all workers).
        # For now, use the last valid experience which is a simpler solution
        self.last_generated_experience = None

        # Those value can be changed
        self.kl_ctl = self.args.kl_ctl
        self.clip_reward_value = 2.5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = self.args.gamma
        self.lam = self.args.lam
        self.generate_time = 0.0
        self.temperature = self.args.temperature
        self.termination_condition = self.args.termination_condition
        self.value_function = self.args.value_function
        self.ngram = getattr(self.args, 'ngram', None) # for fixed n-gram selection
        self.repeat_times = getattr(self.args, 'repeat_times', None) # randomized n-gram selection
        self.parser_cutoff = getattr(self.args, 'parser_cutoff', None) # parsing selection
        

    def _generate_sequence(self, prompts, mask, step):

        max_min_length = self.max_answer_seq_len + prompts.shape[1]

        # This has been added due to a probability/nan error that happens after
        # meta-llama/Llama-2-7b-hf enabled do_sample:
        # https://huggingface.co/meta-llama/Llama-2-7b-hf/commit/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9
        generation_config = dict(do_sample=True, temperature=self.temperature)

        with torch.no_grad():
            if self.args.enable_zero3_generation_gather and self.z3_enabled:
                with unwrap_model_for_generation(self.actor_model) as unwrapped_model:
                    seq = unwrapped_model.generate(
                        prompts,
                        attention_mask=mask,
                        max_length=max_min_length,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        synced_gpus=self.z3_enabled,
                        **generation_config)
            else:
                seq = self.actor_model.module.generate(
                    prompts,
                    attention_mask=mask,
                    max_length=max_min_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    synced_gpus=self.z3_enabled,
                    **generation_config)

        # Filter out seq with no answers (or very short). This happens when users directly use the pre-training ckpt without supervised finetuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        self.prompt_length = prompt_length
        ans = seq[:, prompt_length:]
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)

        if self.args.print_answers and (step % self.args.print_answers_interval
                                        == 0):
            print(
                f"--- prompt --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(prompts, skip_special_tokens=True)}"
            )
            print(
                f"--- ans    --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(ans, skip_special_tokens=True)}"
            )

        out_seq = []
        for i in range(batch_size):
            if valid_ans_len[
                    i] <= 1:  # if the answer is shorter than 1 token, drop it
                print(
                    f'Dropping too short generated answer: {step=}: \n'
                    f'prompts: {self.tokenizer.decode(prompts[i], skip_special_tokens=False)}\n'
                    f'answers: {self.tokenizer.decode(ans[i], skip_special_tokens=False)}'
                )
                continue
            else:
                out_seq.append(seq[i:i + 1])

        if not out_seq:
            print(
                f'All generated results are too short for rank={self.args.local_rank} step={step}\n'
                f'-> prompts: {self.tokenizer.batch_decode(prompts, skip_special_tokens=False)}\n'
                f'-> answers: {self.tokenizer.batch_decode(ans, skip_special_tokens=False)}'
            )
            return None

        out_seq = torch.cat(out_seq, dim=0)  # concat output in the batch dim

        return out_seq

    def generate_experience(self, prompts, mask, step):
        self.eval()
        generate_start = time.time()
        seq = self._generate_sequence(prompts, mask, step)
        generate_end = time.time()
        if seq is None:
            assert self.last_generated_experience is not None, f'Invalid generated experience at {step=}'
            prompts = self.last_generated_experience['prompts']
            seq = self.last_generated_experience['seq']
        else:
            self.last_generated_experience = {'prompts': prompts, 'seq': seq}
        self.train()

        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long()
        with torch.no_grad():
            output = self.actor_model(seq, attention_mask=attention_mask, return_dict=True)
            output_ref = self.ref_model(seq, attention_mask=attention_mask)
            reward_score = self.reward_model.forward_value(
                seq, attention_mask,
                prompt_length=self.prompt_length)['chosen_end_scores'].detach(
                )
            values = self.critic_model.forward_value(
                seq, attention_mask, return_value_only=True).detach()[:, :-1]

        logits = output.logits
        logits_ref = output_ref.logits
        if self.compute_fp32_loss:
            logits = logits.to(torch.float)
            logits_ref = logits_ref.to(torch.float)

        ppl = []
        if self.termination_condition == 'ppl':
            for i in range(prompts.size(1) - 1, logits_ref.size(1)):
                shift_logits = logits_ref[:, :i - 1, :].contiguous().view(-1, self.config.vocab_size)
                shift_labels = seq[:, 1:i, :].contiguous.view(-1)
                loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=pad_token_id)
                ppl.append(torch.exp(loss))

        self.generate_time = generate_end - generate_start

        return {
            'prompts': prompts,
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
            'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:,
                                                                        1:]),
            'value': values,
            'rewards': reward_score,
            'input_ids': seq,
            'attention_mask': attention_mask,
            'ppl': ppl
        }
    
    def parsing_selection(self, start, seq, constituent_tree=None):
        assert self.parser_cutoff is not None, "Please specify the parser cutoff with --parser-cutoff for parsing based selection"
        import re
        def reconstruct_tokenize(leaves):
            # This function is used to migitage the gap between parser tokenizer and policy tokenizer
            tree_tokens = []
            start_pos = 0
            for token in leaves:
                match_pos = sentence.find(token, start_pos)
                before = sentence[match_pos - 1] if match_pos > 0 else ''
                start_pos = match_pos + len(token)
                if before == ' ':
                    tree_tokens.extend(self.tokenizer.tokenize(before + token))
                else:
                    tree_tokens.extend(self.tokenizer.tokenize(token))
            return tree_tokens

        def dfs(cur_tree, passed_tokens):
            leaves = cur_tree.leaves()
            if len(leaves) == 1: # rule 2: nodes with single token is a part of the last macro action
                return False, passed_tokens + len(reconstruct_tokenize(leaves))
            token_of_leaves = len(reconstruct_tokenize(leaves))
            if token_of_leaves < self.parser_cutoff: # rule 1: nodes with fewer than cutoff tokens are treated as a macro action
                local_sequence.append(passed_tokens + token_of_leaves)
                passed_tokens += token_of_leaves
                return True, passed_tokens
            else:
                for nxt_tree in cur_tree:
                    state, _passed_tokens = dfs(nxt_tree, passed_tokens)
                    if state == False:
                        local_sequence[-1] = _passed_tokens
                    passed_tokens = _passed_tokens
                return True, passed_tokens

        sequences = self.tokenizer.batch_decode(seq[:, start + 1:], skip_special_tokens=True)[0]
        sentence_endings = re.compile(r'(?<=[.!?]) +|\n')
        sentences = sentence_endings.split(sequences)
        ends = sentence_endings.findall(sequences)
        ma_lengths = []
        start_offset = 0
        for idx, sentence in enumerate(sentences):
            if sentence == '':
                continue
            passed_tokens = 0
            end = ''
            if idx > 0:
                end = ends[idx - 1]
                
            if '\n' in end:
                passed_tokens == len(self.tokenizer.tokenize(end))
                end = ''

            sentence = end + sentence # tokenizer("\nHello") and tokenizer("Hello") will have different token length
            tokens = self.tokenizer.tokenize(sentence)
            
            tree_dyn = constituent_tree.parse(sentence)
            tree = tree_dyn.nltk_tree
            
            tree_tokens = reconstruct_tokenize(tree.leaves())
            
            # Check: the tokenization of the parser and the policy should be the same, otherwise, fall back to the vanilla PPO
            if len(tokens) != len(tree_tokens):
                raise IndexError("The tokenization of the parser and the policy are not the same.")
    
            local_sequence = [0]
            _, passed_tokens = dfs(tree, passed_tokens)
            if 0 not in local_sequence: # Process the start position
                local_sequence = [0] + local_sequence
            local_ma_lengths = [start + start_offset + item for item in local_sequence]
            start_offset += len(tokens)
            ma_lengths += local_ma_lengths
        # if int(seq.size(1) - 2) < ma_lengths[-1]:
        #     raise IndexError(f'{int(seq.size(1) - 2)} < {ma_lengths[-1]}, {ma_lengths}')
        if int(seq.size(1) - 2) not in ma_lengths:
            ma_lengths.append(int(seq.size(1) - 2))
        ma_lengths = list(set(ma_lengths))
        return ma_lengths
    
    def fixed_ngram_selection(self, start, mask):
        assert self.ngram is not None, "Please specify the n-gram length with --ngram for fixed n-gram selection"
        current_count = 0
        ma_length = [start]
        for i in range(mask[:, start:].size(1) - 1):
            current_count += mask[0, start + i].item()
            if current_count == self.ngram:
                ma_length.append(start + i + 1)
                current_count = 0
        ma_length.append(mask.size(1) - 1)
        return ma_length
    
    def randomized_ngram_selection(self, start, mask):
        assert self.repeat_times is not None, "Please specify the repeat times with --repeat_times for randomized n-gram selection"
        # repeat times cotrols the maximum selected times of a n-gram
        randomized_ngram_list = torch.repeat_interleave(torch.tensor([1, 2, 3, 5, 10], dtype=int), self.repeat_times).tolist()
        import random
        random.shuffle(randomized_ngram_list)
        current_count = 0
        idx = 0
        ma_length = [start]
        for i in range(mask[:, start:].size(1) - 1):
            current_count += mask[0, start + i].item()
            if current_count == randomized_ngram_list[idx]:
                ma_length.append(start + i + 1)
                current_count = 0
                idx += 1
                if idx == len(randomized_ngram_list):
                    break
        # the rest of the tokens, which is not modeled as macro action, are treated as a macro action with length \infty
        ma_length.append(mask.size(1) - 1)
        return ma_length
    
    def ppl_selection(self, start, mask, ppl):
        assert ppl is not None
        ma_length = [start]
        for i in range(len(ppl) - 1):
            if ppl[i + 1] > ppl[i]:
                ma_length.append(start + i)
        ma_length.append(mask.size(1) - 1)
        return ma_length
            
    def modeling_macro_action(self, variables, mask, start, ma_lengths):
        split_list = torch.diff(torch.tensor(ma_lengths)).tolist()
        split_list += [1]
        
        splited_vars = torch.split(variables[:, start:], split_list, dim=-1)
        splited_mask = torch.split(mask[:, start:], split_list, dim=-1)
        
        ma_variables = torch.zeros(variables.size(0), len(split_list)).to(variables)
        
        for idx, (var_i, mask_i) in enumerate(zip(splited_vars, splited_mask)):
            masked_vars = var_i[mask_i != 0]

            if self.value_function == 'equal':
                sigma = torch.ones_like(masked_vars) / masked_vars.numel()
            elif self.value_function == 'unit':
                sigma = torch.zeros_like(masked_vars)
                sigma[:, -1] = 1.0
            elif self.value_function == 'position':
                sigma = 1.0 / (torch.arange(masked_vars.numel()) + 1)
                sigma = sigma / torch.sum(sigma)
            sigma = sigma.to(masked_vars)
            ma_variables[:, idx] = torch.sum(masked_vars * sigma) if masked_vars.numel() > 0 else 0.0
        return ma_variables

    def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score, action_mask, return_kl = False):
        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = torch.clone(kl_divergence_estimate)
        start = prompts.shape[1] - 1
        ends = start + action_mask[:, start:].sum(1) + 1
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value, self.clip_reward_value)

        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j]

        if return_kl:
            return rewards, kl_divergence_estimate
        return rewards

    def train_rlhf(self, inputs, constituent_tree=None):
        # train the rlhf mode here
        ### process the old outputs
        prompts = inputs['prompts']
        log_probs = inputs['logprobs']
        ref_log_probs = inputs['ref_logprobs']
        reward_score = inputs['rewards']
        values = inputs['value']
        attention_mask = inputs['attention_mask']
        seq = inputs['input_ids']
        ppl = inputs['ppl']
        
        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:, 1:]
        
        inputs['resp_length'] = torch.tensor(seq[:, start:].size(1)).to(seq)

        old_values = values

        with torch.no_grad():
            old_rewards, kl_divergence = self.compute_rewards(prompts, log_probs,
                                               ref_log_probs, reward_score,
                                               action_mask, return_kl=True)
            ends = start + action_mask[:, start:].sum(1) + 1
            # we need to zero out the reward and value after the end of the conversation
            # otherwise the advantage/return will be wrong
            for i in range(old_rewards.shape[0]):
                old_rewards[i, ends[i]:] = 0
                old_values[i, ends[i]:] = 0
                
            # obtain the macro action lengths. 
            if self.termination_condition == 'fixed':
                ma_lengths = self.fixed_ngram_selection(start, action_mask)
            elif self.termination_condition == 'randomized':
                ma_lengths = self.randomized_ngram_selection(start, action_mask)
            elif self.termination_condition == 'ppl':
                ma_lengths = self.ppl_selection(start, action_mask, ppl)
            elif self.termination_condition == 'parsing':
                ma_lengths = self.parsing_selection(start, seq, constituent_tree)
            # calculate MA_values and MA_rewards
            macro_action_values = self.modeling_macro_action(old_values, action_mask, start, ma_lengths)
            # NOTE: when modeling macro action rewards, we need to guarantee that the length of the last macro action is equal to 1, otherwise the rewards will also involved in calculation.
            # TODO: another solution is we can apply this function before the KL divengence is summed up with rewards, and operate on the KL divergence.
            macro_action_rewards = self.modeling_macro_action(old_rewards, action_mask, start, ma_lengths)
            
            advantages, returns = self.get_advantages_and_returns(macro_action_values, macro_action_rewards, 0)

        ### process the new outputs
        batch = {'input_ids': seq, "attention_mask": attention_mask}
        actor_prob = self.actor_model(**batch, use_cache=False).logits
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])
        actor_loss = self.actor_loss_fn(actor_log_prob[:, start:],
                                        log_probs[:, start:], advantages,
                                        action_mask[:, start:], ma_lengths)
        self.actor_model.backward(actor_loss)

        if not self.args.align_overflow:
            self.actor_model.step()

        value = self.critic_model.forward_value(**batch,
                                                return_value_only=True,
                                                use_cache=False)[:, :-1]
        critic_loss = self.critic_loss_fn(value[:, start:], old_values[:,
                                                                       start:],
                                          returns, action_mask[:, start:], ma_lengths)
        self.critic_model.backward(critic_loss)

        if self.args.align_overflow:
            actor_overflow = self.actor_model.optimizer.check_overflow(
                external=True)
            critic_overflow = self.critic_model.optimizer.check_overflow(
                external=True)

            rank = torch.distributed.get_rank()
            if actor_overflow and not critic_overflow:
                self.critic_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: actor overflow, skipping both actor and critic steps",
                    rank)
            elif not actor_overflow and critic_overflow:
                self.actor_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: critic overflow, skipping both actor and critic steps",
                    rank)
            elif actor_overflow and critic_overflow:
                print_rank_0(
                    "OVERFLOW: actor and critic overflow, skipping both actor and critic steps",
                    rank)
            self.actor_model.step()

        self.critic_model.step()

        return actor_loss, critic_loss, kl_divergence

    def get_overflow(self):
        # Overflow is not expected when using bf16
        # Therefore, DeepSpeed's BF16_Optimizer does not maintain an overflow indication
        if self.args.dtype == "bf16":
            return False, False

        actor_overflow = self.actor_model.optimizer.overflow
        critic_overflow = self.critic_model.optimizer.overflow

        return actor_overflow, critic_overflow

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask, ma_lengths):
        ## policy gradient loss
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)

        split_list = torch.diff(torch.tensor(ma_lengths)).tolist()
        split_list += [1]
        
        split_ratio = torch.split(ratio, split_list, dim=-1)
        split_mask = torch.split(mask, split_list, dim=-1)

        pg_loss = 0.0
        total_mask_sum = 0.0
        for i in range(len(split_list)):
            ratio_i = split_ratio[i]
            mask_i = split_mask[i]
            advantages_i = advantages[:, i]
            
            pg_loss1 = -advantages_i * ratio_i
            pg_loss2 = -advantages_i * torch.clamp(ratio_i, 1.0 - self.cliprange, 1.0 + self.cliprange)

            pg_loss += torch.sum(torch.max(pg_loss1, pg_loss2) * mask_i)
            total_mask_sum += mask_i.sum()
        
        pg_loss = pg_loss / total_mask_sum
        return pg_loss


    def critic_loss_fn(self, values, old_values, returns, mask, ma_lengths):
        ## value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        if self.compute_fp32_loss:
            values = values.float()
            values_clipped = values_clipped.float()

        split_list = torch.diff(torch.tensor(ma_lengths)).tolist()
        split_list += [1]

        splited_values = torch.split(values, split_list, dim=-1)
        splited_values_clipped = torch.split(values_clipped, split_list, dim=-1)
        splited_mask = torch.split(mask, split_list, dim=-1)

        total_vf_loss = 0.0
        total_mask_sum = 0.0

        # Iterate over each split segment and calculate the loss
        for i in range(len(splited_values)):
            vf_loss1 = (splited_values[i] - returns[:, i])**2
            vf_loss2 = (splited_values_clipped[i] - returns[:, i])**2
            vf_loss = 0.5 * torch.sum(
                torch.max(vf_loss1, vf_loss2) * splited_mask[i])
            total_vf_loss += vf_loss
            total_mask_sum += splited_mask[i].sum()
        
        total_vf_loss = total_vf_loss / total_mask_sum
        return total_vf_loss


    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns

    def _validate_training_mode(self):
        assert self.actor_model.module.training
        assert self.critic_model.module.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.module.training
        assert not self.critic_model.module.training
        assert not self.ref_model.module.training
        assert not self.reward_model.module.training

    def train(self):
        self.actor_model.train()
        self.critic_model.train()

    def eval(self):
        self.actor_model.eval()
        self.critic_model.eval()
        self.reward_model.eval()
        self.ref_model.eval()

    def dump_model_norms(self, tag):
        actor_model_norm = get_model_norm(self.actor_model)
        ref_model_norm = get_model_norm(self.ref_model)
        critic_model_norm = get_model_norm(self.critic_model)
        reward_model_norm = get_model_norm(self.reward_model)
        print_all_ranks(f'{tag} global_actor_model_norm', actor_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_ref_model_norm', ref_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_critic_model_norm', critic_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_reward_model_norm', reward_model_norm,
                        self.args.local_rank)


class DeepSpeedPPOTrainerUnsupervised(DeepSpeedPPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()

        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
