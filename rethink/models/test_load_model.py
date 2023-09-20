from rethink.models import create_model


arch = "yolov7"

print(
    "\n\n\n==== YOLOV7 ==== \n\n"
    + "missing_keys=['linear_layers.2.weight', 'linear_layers.2.bias', 'linear_layers.5.weight', 'linear_layers.5.bias'],\n"
    + "unexpected_keys=['backbone.model.51.cv1.conv.weight,...'\n"
    + "============\n\n"
)


m = create_model(arch, 0)

print(
    "\n\n\n==== YOLOV7 Custom ====\n\n"
    + "missing_keys=['linear_layers.2.weight', 'linear_layers.2.bias', 'linear_layers.5.weight', 'linear_layers.5.bias'],\n"
    + "unexpected_keys=['backbone.model.51.cv1.conv.weight,...'\n"
    + "============\n\n"
)

weights = "/media/magalhaes/vision/checkpoints/yolov7_1681719082/weights/best.pt"
m = create_model(arch, 0, weights)

print(
    "\n\n\n==== YOLOV7 tiny ==== \n\n"
    + "missing_keys=['linear_layers.2.weight', 'linear_layers.2.bias', 'linear_layers.5.weight', 'linear_layers.5.bias'],\n"
    + "unexpected_keys=['backbone.model.29.conv.weight,...'\n"
    + "============\n\n"
)


arch = "yolov7_tiny"

m = create_model(arch, 0)

print("\n\n\n==== YOLOV7 tiny finetuned ====\n\n")

arch = "yolov7_tiny"
weights = "/media/cache/magalhaes/sound/checkpoints/online_teacher_yolov7_tiny_1680617732_1680684508/model_best_1.pth"
m = create_model(arch, 7, weights)

print("\n\n\n==== YOLOV7 distilled ====\n\n")


arch = "yolov7_tiny_distilled"

weights = "/media/cache/magalhaes/sound/distillations/online_teacher_yolov7_tiny_1680619925/last0.pth"

m = create_model(arch, 0, weights)
