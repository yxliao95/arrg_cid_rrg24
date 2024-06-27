class exp4_vis2lan:
    seed = 0
    target_report_section = "impression"  # findings, impression

    # The final dir path will be `output_xx_dir/output_name/`
    output_name = "baseline"
    output_result_dir = "/home/liao/workspace/arrg_prototype/outputs/results"
    output_model_dir = "/home/liao/workspace/arrg_prototype/outputs/models"
    output_checkpoint_dir = "/home/liao/workspace/arrg_prototype/outputs/checkpoints"

    data_path = {
        "mimic": "/home/liao/data/mimic-cxr-jpg-resized",
        "interpret": "/home/liao/data/interpret-cxr",
        "interpret-test-public": "/home/liao/data/interpret-cxr-test-public",
        "interpret-test-hidden": "/home/liao/data/interpret-cxr-test-hidden",
        "img_labels": "/home/liao/data/img_label_ids_5000.json",
        "img_label_string": "/home/liao/data/label_map_5000.json",
    }

    model_name_or_path_dict = {
        "roberta-base": "FacebookAI/roberta-base",
        "swinv2-base": "microsoft/swinv2-base-patch4-window8-256",
    }

    vision_model_path = model_name_or_path_dict["swinv2-base"]
    language_model_path = model_name_or_path_dict["roberta-base"]

    img_label_present = 80
    img_label_absent = 23
    img_label_uncertain = 11

    do_train = True
    do_inference = False
    load_from_pretrained = False  # "model_dir" or False or ""
    load_from_resume = False  # "checkpoint_dir" or False or ""

    resume_training = False  # We don't support resume training for now.

    # Dataloader
    max_seq_len = 512  # If None, then use the model max valid seq len (512+2)
    train_batch_size = 12
    eval_batch_size = 24

    # train
    lr = 1e-4
    mlc_lr = 1e-4
    weight_decay = 0.1
    warmup_proportion = 0.1
    image_loss_weight = 1
    text_loss_weight = 1

    clip_grad_norm = 1  # ignored when set to None
    grad_accum_steps = 1
    num_epoch = 30

    # steps means the optimization steps
    logging_steps = 100
    # 0 to disable
    eval_epoch = 1
    eval_steps = 20000  # save model at the best eval result
    save_epoch = 1  # save checkpoint to resume

    # eval
    max_generation_len = 128
    num_beam = 3
    topk = 10
    threshold = 0.5
