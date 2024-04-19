import time


def bench_inference(tokenizer, model, prompt, generate_kwargs, num_iter, num_warmup):
    total_time = 0.0
    for i in range(num_iter):
        tic = time.time()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output = model.generate(input_ids, **generate_kwargs)
        gen_ids = output
        gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        toc = time.time()
        input_tokens_lengths = [x.shape[0] for x in input_ids]
        output_tokens_lengths = [x.shape[0] for x in gen_ids]
        total_new_tokens = [
            o - i
            for i, o in zip(input_tokens_lengths, output_tokens_lengths)
        ]
        print(gen_text, total_new_tokens, flush=True)
        print("Iteration: %d, Time: %.6f sec" % (i, toc - tic), flush=True)
        if i >= num_warmup:
            total_time += toc - tic
        return total_time