from pathlib import Path
import onnx
from onnx import helper, TensorProto
import pytest
from maps.graph import import_onnx_model, run_graph_rewrites
from maps.hardware import WorkSignature, WorkKind
from maps.target import magia_v3
from maps.deployment import build_application, validate_application

def model(path, shape=(1, 4, 112, 112), output=(1, 4, 56, 56), **attrs):
    attrs = {"kernel_shape": [3,3], "strides": [2,2], "pads": [1,1,1,1], **attrs}
    graph=helper.make_graph([helper.make_node("MaxPool",["x"],["y"],**attrs)],"pool",
        [helper.make_tensor_value_info("x",TensorProto.FLOAT,shape)],
        [helper.make_tensor_value_info("y",TensorProto.FLOAT,output)])
    if "auto_pad" in attrs:
        del graph.node[0].attribute[:]
        attrs.pop("pads",None)
        graph.node[0].attribute.extend(helper.make_attribute(k,v) for k,v in attrs.items())
    onnx.save(helper.make_model(graph,opset_imports=[helper.make_opsetid("",20)]),path)
    return path

def test_generated_pool_uses_spatz_and_preserves_geometry(tmp_path):
    path=model(tmp_path/"pool.onnx")
    app=build_application(path,tmp_path/"app",target="magia-v3",mesh_width=2,mesh_height=2,num_token_slots=2)
    assert validate_application(app)["tasks"]==["maxpool2d_fp16_spatz_task"]
    assert "MAPS_HAS_MAXPOOL_SPATZ_TASK=1" in (app/"CMakeLists.txt").read_text()
    assert "runtime.maxpool_fp16_task = MAXPOOL2D_FP16_SPATZ_TASK" in (app/"src/pool_runner.c").read_text()
    tiles="".join(p.read_text() for p in (app/"src/tiles").glob("*.c"))
    assert "OP_MAX_POOL" in tiles
    assert ".params = {3u, 3u, 2u, 2u, 1u, 1u, 1u, 1u}" in tiles

@pytest.mark.parametrize("attrs,shape,output",[
    ({"kernel_shape":[2,3],"strides":[2,2],"pads":[0,1,1,0],"ceil_mode":1},(2,3,5,7),(2,3,3,4)),
    ({"auto_pad":"SAME_LOWER","pads":[0,0,0,0]},(1,3,8,8),(1,3,4,4)),
])
def test_pool_geometry_and_channel_slices(tmp_path,attrs,shape,output):
    imported=import_onnx_model(model(tmp_path/"pool.onnx",shape,output,**attrs))
    mesh=magia_v3.build_mesh(width=2,height=2)
    spec=magia_v3.specialize(run_graph_rewrites(imported),mesh).model
    node=spec.graph.nodes[0]
    assert WorkSignature.from_node(node).work_kind is WorkKind.MAX_POOL
    assert mesh.tiles[0].assigned_device(WorkSignature.from_node(node)).name=="core"
    from maps.planning.mapping import Submesh
    submesh=Submesh(mesh=mesh,submesh_id=0,x0=0,y0=0,width=2,height=2)
    payload=node.payload
    work=payload.build_tile_work(payload.output_layouts(submesh),mesh.tiles[0])
    assert work.input_slice.dims[2].length==shape[2]
    assert work.input_slice.dims[1]==work.output_slice.dims[1]

def test_pool_rejects_dilations_explicitly(tmp_path):
    with pytest.raises(NotImplementedError,match="unit dilation"):
        import_onnx_model(model(tmp_path/"pool.onnx",output=(1,4,55,55),dilations=[2,2]))


def test_complete_resnet_has_device_implementations():
    path=next((Path(__file__).resolve().parents[3]/"maps-experiments/workloads/resnet/model").glob("*.onnx"),None)
    if path is None:
        pytest.skip("Sibling ResNet experiment model is unavailable")
    mesh=magia_v3.build_mesh(width=1,height=1)
    spec=magia_v3.specialize(run_graph_rewrites(import_onnx_model(path)),mesh).model
    for node in spec.graph.nodes:
        mesh.tiles[0].assigned_device(WorkSignature.from_node(node))
    assert sum(WorkSignature.from_node(n).work_kind is WorkKind.MAX_POOL for n in spec.graph.nodes)==1


def test_maxpool_cost_tracks_measured_resnet_shard():
    from types import SimpleNamespace
    work=SimpleNamespace(work_kind=WorkKind.MAX_POOL,operation_count=lambda:2*56*56*9)
    assert magia_v3.CORE_DEVICE.cycles(work)==pytest.approx(1016190.5,rel=.02)
