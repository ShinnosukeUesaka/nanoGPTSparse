# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-enwik8-char-sparse'
eval_iters = 100
eval_interval = 500
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'enwik8-char-sparse'
wandb_run_name = 'mini-gpt-sparse'

dataset = 'enwik8_char'
gradient_accumulation_steps = 2
#batch_size = 64
batch_size = 32
block_size = 1024 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 9
n_head = 8
n_embd_a = 512
n_embd_b = 512
n_embd_attention = 512
true_mask = True

warmup_iters = 100 # not super necessary potentially

# on macbook also add
device = 'cuda'  # run on cpu only
compile = False # do not torch compile the model

sparse_model = True