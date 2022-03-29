"""
Модуль включает в себе функции для построения движков моделей на TensorRT на базе моделей формата onnx
"""
import argparse
import tensorrt as trt

TRT_LOGGER = trt.Logger()


def build_engine(model_path):
    """
    Построение движка (engine) модели формата TensorRT от
    Args:
        model_path: путь к модели в формате onnx
    """
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch) as network,\
            builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config.max_workspace_size = 1 << 30
        builder.max_batch_size = 1

        with open(model_path, 'rb') as model:
            parser.parse(model.read())

        last_layer = network.get_layer(network.num_layers - 1)
        print(network)
        network.mark_output(last_layer.get_output(0))
        
        # Check if last layer recognizes it's output
        if not last_layer.get_output(0):
            # If not, then mark the output using TensorRT API
            print('TEST!!!')
            network.mark_output(last_layer.get_output(0))

        engine = builder.build_engine(network, config)
        engine_path = model_path.replace('.onnx', '.engine')
        print(model_path, engine_path)
        save_engine(engine, engine_path)


def save_engine(engine, file_name):
    """
    Сохранение движка (engine) модели на TensorRT по указанному пути
    Args:
        engine: движок модели на TensorRT
        file_name: название файла для сохранения
    """
    buf = engine.serialize()
    with open(file_name, 'wb') as f:
        f.write(buf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../weights/ga_model.onnx', help='model.onnx path')
    opt = parser.parse_args()
    
    build_engine(opt.model)
