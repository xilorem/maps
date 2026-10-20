"""Packed model weights must not consume the separate V3 runtime arena."""
import json
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from maps.deployment import build_application, validate_application
from maps.target import magia_v3


def test_weights_larger_than_runtime_arena_use_bulk_memory(tmp_path):
    shape=(4096,2048)
    graph=helper.make_graph(
        [helper.make_node('Gemm',['x','w'],['y'],transB=1)],'bulk_weights',
        [helper.make_tensor_value_info('x',TensorProto.FLOAT16,[1,shape[1]])],
        [helper.make_tensor_value_info('y',TensorProto.FLOAT16,[1,shape[0]])],
        [numpy_helper.from_array(np.zeros(shape,dtype=np.float16),name='w')])
    path=tmp_path/'weights.onnx'
    onnx.save(helper.make_model(graph,opset_imports=[helper.make_opsetid('',20)]),path)
    app=build_application(path,tmp_path/'app',target='magia-v3',name='bulk_weights',mesh_width=8,mesh_height=8,num_token_slots=2)
    manifest=validate_application(app)
    memory=manifest['memory']
    assert memory['initializers_bytes']>magia_v3.L2_ARENA_END-magia_v3.L2_ARENA_BASE
    assert memory['initializers_region']=='l2_bulk'
    assert memory['required_l2_bytes']-memory['initializers_bytes']<65536
    assert '.l2_bulk.maps_initializers' in (app/'src/bulk_weights_initializers.S.in').read_text()


def test_runtime_buffers_still_obey_their_own_arena_limit(tmp_path):
    import pytest

    shape=[4_194_304]  # 8 MiB each for input and output; runtime arena is 15 MiB.
    graph=helper.make_graph([helper.make_node('Relu',['x'],['y'])], 'large_runtime',
        [helper.make_tensor_value_info('x',TensorProto.FLOAT16,shape)],
        [helper.make_tensor_value_info('y',TensorProto.FLOAT16,shape)])
    model=tmp_path/'runtime.onnx'
    onnx.save(helper.make_model(graph,opset_imports=[helper.make_opsetid('',20)]),model)
    with pytest.raises(RuntimeError,match='generation failed'):
        build_application(model,tmp_path/'app',target='magia-v3',mesh_width=8,mesh_height=8,num_token_slots=2)
    failures=list(tmp_path.glob('app.generation-failure-*'))
    assert len(failures)==1
    error=json.loads((failures[0]/'backend-error.json').read_text())
    assert 'L2 runtime allocation' in error['stderr']
