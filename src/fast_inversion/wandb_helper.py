def parse_results_to_wandb_log_format(results):
    new_res = {}
    for key, value in results.items():
        value = value[-1]
        prefixes = ['best', 'train', 'val', 'test']
        for prefix in prefixes:
            if prefix in key:
                prefix = f'{prefix}/{key}'
                break
        new_res[key] = value

    return new_res


def init_wandb(args):
    if not args.use_wandb:
        return None
    try:
        import wandb
    except:
        raise ImportError('WandB is not installed.')
    # if cfg.wandb.name == '':
    #     wandb_name = make_wandb_name(cfg)
    # else:
    #     wandb_name = cfg.wandb.name
    try:
        run = wandb.init(entity='daniel-ai', project=args.exp_name,
                         name='')
    except Exception:
        wandb.login(key=get_wandb())
        run = wandb.init(entity='daniel-ai', project=args.exp_name,
                         name='')
    run.config.update(args)
    return run
