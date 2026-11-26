"""RP1 with one physical hexahedral layer and translated 3-D operator images."""
import argparse
import fcntl
from functools import partial
import json
import os
from pathlib import Path
import resource
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / "data/out/NCFV/single_layer_barth"
import runtime as rt
sys.path.insert(0, str(HERE.parent / "riemann_barth_400x400x5"))
import generate_mesh as meshes
from generate_dual_mean_initial import write_case
import h5py
import numpy as np

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def read(path):
    return json.loads(path.read_text())


def prepare(n, copies):
    physical = OUT / f"inputs_{n}"
    directory = OUT / f"operator_images/{n}_{copies}"
    directory.mkdir(parents=True, exist_ok=True)
    mesh, csv = physical / "grid.cgns", physical / "RP1.csv"
    if not mesh.exists():
        meshes.write_mesh(mesh, n, n, 1, 1/n)
    if not csv.exists():
        write_case("RP1", csv, mesh, rt.digest(mesh), (n, n, 1, 1/n), (.8, .8))
    image_mesh, image_csv = directory / "grid.cgns", directory / "RP1.csv"
    if (directory / "manifest.json").exists():
        return
    meta = read(csv.with_suffix(".metadata.json"))
    assert meta["discontinuity_xy"] == [.8, .8] and meta["cells_per_axis"] == [n, n, 1]
    assert rt.digest(mesh) == meta["mesh_sha256"] and rt.digest(csv) == meta["csv_sha256"]
    if not image_mesh.exists():
        meshes.write_mesh(image_mesh, n, n, copies, copies / n)
    # Verify every auxiliary hex is a pure translate of a physical input hex.
    with h5py.File(mesh) as one, h5py.File(image_mesh) as many:
        group = "Base/blk-1/"
        path = group + "HexElements/ElementConnectivity/ data"
        cells = one[path][:].reshape(-1, 8)
        plate = (n + 1)**2
        for k in range(copies):
            np.testing.assert_array_equal(many[path][k*cells.size:(k+1)*cells.size].reshape(-1, 8),
                                          cells + k * plate)
        for axis in "XYZ":
            path = group + f"GridCoordinates/Coordinate{axis}/ data"
            ref = one[path][:plate]
            for k in range(copies+1):
                np.testing.assert_allclose(many[path][k*plate:(k+1)*plate], ref + (k/n if axis == "Z" else 0),
                                           rtol=0, atol=5e-15)
    with csv.open("rb") as source:
        header = source.readline()
        assert header.strip() == b"globalNode,rho,rhou,rhov,rhow,rhoE"
        rows = []
        for k in range(2*plate):
            key, values = source.readline().split(b",", 1)
            assert int(key) == k
            if k < plate:
                rows.append(values)
            else:
                assert rows[k-plate] == values
        assert not source.read(1)
    temp = image_csv.with_suffix(".partial")
    with temp.open("wb", buffering=1024*1024) as out:
        out.write(header)
        for z in range(copies+1):
            for k, value in enumerate(rows):
                out.write(str(k+plate*z).encode()+b","+value)
    temp.replace(image_csv)
    rt.save(directory/"manifest.json", dict(physical_mesh=str(mesh), physical_mesh_sha256=meta["mesh_sha256"],
        physical_field_sha256=meta["csv_sha256"], physical_cells=[n,n,1], physical_period=1/n,
        operator_image_periods=copies, mesh_sha256=rt.digest(image_mesh), field_sha256=rt.digest(image_csv)))


def run(folder,n,copies,ranks,end,*,restart=None,uniform=False,local_barth=False,fields=None,test_dt=1e-4):
    physical = OUT/f"inputs_{n}"
    images = OUT/f"operator_images/{n}_{copies}"
    prepare(n,copies)
    cfg={"/mesh/meshFile":str(physical/"grid.cgns"), "/mesh/periodicLengths":[0,0,1/n],
         "/initialField/nodeFile":str(physical/"RP1.csv"), "/time/endTime":end}
    env=dict(NCFV_P1_BOTH_SIDES="1",NCFV_P1_COMPONENTWISE="0",NCFV_P1_SLOPE_LIMITER="Barth",
             NCFV_P1_FLUX="Roe",NCFV_FLUX_COMPONENTWISE="0",NCFV_FLUX_RELAXATION="2.5e-11",
             NCFV_LAYER_IMAGES=str(copies), NCFV_LAYER_SIDE=str(n+1),
             NCFV_LAYER_IMAGE_MESH=str(images/"grid.cgns"),NCFV_LAYER_IMAGE_FIELD=str(images/"RP1.csv"),
             NCFV_CELL_AUDIT_INTERVAL="100" if n==800 else "1", NCFV_BAR_AUDIT_INTERVAL="100" if n==800 else "1",
             NCFV_BAR_AUDIT_FILE=str(folder/"bar_audit.jsonl"))
    if local_barth:
        env["NCFV_P1_LOCAL_COMPONENTWISE"]="1"
    if fields is not None:
        cfg["/initialField/nodeFile"],env["NCFV_LAYER_IMAGE_FIELD"]=map(str,fields)
    if n!=800:
        cfg.update({"/time/maximumTimeStep":test_dt,"/time/reportInterval":1,"/io/restartInterval":0})
    if uniform:
        cfg.update({"/initialField/nodeFile":"","/physics/initialPrimitive":[1.2,.1,-.2,.35,1.]})
        env["NCFV_LAYER_IMAGE_FIELD"]=""
    if restart:
        cfg["/io/restartInput"]=str(restart).removesuffix(".dnds.h5")
    rt.run(folder,ranks,cfg,extra_env=env,poll_seconds=6 if n==800 else .15)
    chk=max(folder.glob("restart_*.dnds.h5"));u,p,t,it=rt.read_checkpoint(chk)
    assert abs(t-end)<1e-12 and u.shape==((n+1)**2,5)
    with h5py.File(chk) as f:
        conf=json.loads(f.attrs["configuration"])
        assert conf["mesh"]["meshFile"]==str(physical/"grid.cgns") and conf["mesh"]["periodicLengths"]==[0,0,1/n]
        assert int(f.attrs["operatorImagePeriods"])==copies
        assert bool(f.attrs.get("localComponentwiseBarth", 0)) == local_barth
    audits=[json.loads(l) for l in (folder/"cell_audit.jsonl").read_text().splitlines()]
    bars=[json.loads(l) for l in (folder/"bar_audit.jsonl").read_text().splitlines()]
    assert audits and bars and all(a["method"]=="BARTH_P1_BOTH_SIDES" and not a["componentwise_p1"]
                    and not a["fallback_llf"] and a["p1_slope_limiter"]==0 for a in audits)
    assert all(a.get("local_componentwise_p1",False)==local_barth and
               not a.get("componentwise_flux",False) for a in audits)
    with (folder/"solver.log").open() as log:
        geometry=[l.strip() for l in log if 'Periodic single layer:' in l or 'max closure=' in l]
    assert len(geometry)==2
    result=dict(time=t,step=it,checkpoint=str(chk),physical_nodes=len(u),geometry=geometry,
        rho_range=[float(u[:,0].min()),float(u[:,0].max())],pressure_range=[float(p.min()),float(p.max())],
        max_abs_w=float(abs(u[:,3]/u[:,0]).max()),samples=len(audits),mean_defect=max(a["maximum_mean_defect"] for a in audits),
        image_limited_face_samples=sum(a["limited_faces"] for a in audits),
        image_violating_limited_face_samples=sum(a["violating_limited_faces"] for a in audits),
        nonphysical=max(a["nonphysical_traces"] for a in audits),invalid_bar=max(b["invalid_corrected_bar_states"] for b in bars),
        max_image_integrated_flux_balance=max(abs(x) for b in bars for x in b["conservation_defect"]))
    if n==800:
        assert read(folder/"run_status.json")["physical_binding_verified"]
    if local_barth:
        result["local_barth"]=dict(
            gated_node_samples=sum(a["local_p1_gated_nodes"] for a in audits),
            full_weight_node_samples=sum(a["local_p1_full_weight_nodes"] for a in audits),
            changed_fallback_node_samples=sum(a["local_p1_changed_fallback_nodes"] for a in audits),
            total_image_node_samples=sum(a["total_nodes"] for a in audits))
    rt.save(folder/"result.json",result)
    return u,result


def validate(local_barth=False):
    root=OUT/"local_barth" if local_barth else OUT
    d=root/"validation"/rt.digest(rt.EXE)[:16]
    d.mkdir(exist_ok=True,parents=True)
    solve=partial(run,local_barth=local_barth)
    constant,r0=solve(d/"constant",20,7,4,.001,uniform=True)
    u0=np.array([1.2,.12,-.24,.42,2.5+.6*(.1**2+.2**2+.35**2)])
    err_constant=float(abs(constant-u0).max())
    serial,rs=solve(d/"serial7",20,7,1,.001)
    multi,rm=solve(d/"mpi7",20,7,4,.001)
    nine,r9=solve(d/"mpi9",20,9,4,.001)
    _,half=solve(d/"restart_seed",20,7,4,.0005)
    resumed,rr=solve(d/"restart_np2",20,7,2,.001,restart=half["checkpoint"])
    errors=dict(uniform_state=err_constant,serial_vs_four=float(abs(serial-multi).max()),
                seven_vs_nine_images=float(abs(multi-nine).max()),restart_repartition=float(abs(resumed-multi).max()))
    print("validation",errors,flush=True)
    assert err_constant<2e-13 and max(errors.values())<1e-8
    assert all(r["mean_defect"]<1e-12 and r["nonphysical"]==r["invalid_bar"]==0 and
               r["max_image_integrated_flux_balance"]<1e-12 for r in (r0,rs,rm,r9,rr))
    if local_barth:
        assert r0["local_barth"]["gated_node_samples"]==0
        validate_rp3(d)
    rt.save(root/"validation.json",dict(errors=errors, test_time=.001, runs=[r0,rs,rm,r9,rr],
                                       validation_directory=str(d)))


def validate_rp3(directory):
    fields=[]
    for nz,mesh in [(1,OUT/"inputs_20/grid.cgns"),(7,OUT/"operator_images/20_7/grid.cgns")]:
        csv=directory/f"rp3_inputs_{nz}/RP3.csv"
        meta=csv.with_suffix(".metadata.json")
        if not csv.exists():
            write_case("RP3",csv,mesh,rt.digest(mesh),(20,20,nz,nz/20),(.5,.5))
        info=read(meta)
        assert info["case"]=="RP3" and info["discontinuity_xy"]==[.5,.5] and info["csv_sha256"]==rt.digest(csv)
        fields.append(csv)
    control,_=run(directory/"rp3_shared",20,7,4,.3,fields=fields,test_dt=.001)
    local,result=run(directory/"rp3_local",20,7,4,.3,fields=fields,local_barth=True,test_dt=.001)
    counts=result["local_barth"]
    assert 0 < counts["changed_fallback_node_samples"] <= counts["gated_node_samples"] < counts["total_image_node_samples"]
    assert result["image_violating_limited_face_samples"]==result["nonphysical"]==result["invalid_bar"]==0
    assert result["mean_defect"] < 1e-12
    difference=float(np.max(np.abs(local-control)))
    assert difference>0
    rt.save(directory/"rp3_comparison.json",dict(time=result["time"],max_conservative_difference=difference,**counts))


def finish(folder=OUT/"full", output=OUT/"RP1_t05_density_pressure.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    result=read(folder/"result.json");u,p,t,_=rt.read_checkpoint(Path(result["checkpoint"]))
    x=np.linspace(0,1,801)
    fig,axes=plt.subplots(1,2,figsize=(11.2,5.4),layout="constrained")
    for ax,q,name in zip(axes,(u[:,0],p),("Density","Pressure")):
        field=q.reshape(801,801)
        im=ax.pcolormesh(x,x,field,cmap="coolwarm",shading="nearest",rasterized=True)
        levels=np.linspace(q.min(),q.max(),27)[1:-1]
        ax.contour(x,x,field,levels=levels,colors="black",linewidths=.35)
        ax.set(xlim=(0,1),ylim=(0,1),aspect="equal",title=name,xlabel="x",ylabel="y")
        fig.colorbar(im,ax=ax,shrink=.82)
    label="local Barth P1" if "local_barth" in result else "shared Barth P1"
    fig.suptitle(f"RP1  |  800 x 800 x 1  |  {label}  |  initial split (0.8, 0.8)  |  t = {t:g}")
    fig.savefig(output,dpi=175);plt.close(fig)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("action",choices=["validate","full","finish","extend"])
    ap.add_argument("--restart",type=Path)
    ap.add_argument("--until",type=float)
    ap.add_argument("--exe",type=Path,default=rt.EXE,help="Path to the single-period solver from build.sh")
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument("--local-barth",dest="local_barth",action="store_true",
                      help="Use the planar-compression limiter (default for fresh runs)")
    mode.add_argument("--shared-barth",dest="local_barth",action="store_false",
                      help="Use the original shared-slope limiter")
    ap.set_defaults(local_barth=None)
    args=ap.parse_args()
    rt.EXE=args.exe.absolute()
    action=args.action
    if action=="extend" and (args.restart is None or args.until is None):
        ap.error("extend requires --restart and --until")
    local_barth=True if args.local_barth is None else args.local_barth
    if action=="extend":
        source=args.restart.resolve(strict=True)
        status=source.parent/"run_status.json"
        source_inputs=read(status)["inputs"] if status.is_file() else None
        with h5py.File(source) as checkpoint:
            source_mode=checkpoint.attrs.get("localComponentwiseBarth")
        if source_mode is None and source_inputs is not None:
            source_mode=source_inputs["extra_env"].get("NCFV_P1_LOCAL_COMPONENTWISE")=="1"
        if args.local_barth is None:
            local_barth=bool(source_mode)
    elif action=="finish" and args.local_barth is None:
        local_barth=(OUT/"local_barth/full/result.json").is_file()
    study_root=OUT/"local_barth" if local_barth else OUT
    study_root.mkdir(exist_ok=True,parents=True)
    with (OUT/"study.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if action=="validate":
            validate(local_barth)
        elif action=="full":
            validate(local_barth)
            run(study_root/"full",800,9,256,.5,local_barth=local_barth)
            finish(study_root/"full",study_root/"RP1_t05_density_pressure.png")
        elif action=="extend":
            with h5py.File(source) as f:
                start=float(f.attrs["simulationTime"])
                step=int(f.attrs["iteration"])
                assert f["state/origIndex"].size==801**2 and int(f.attrs["operatorImagePeriods"])==9
            if not np.isfinite(args.until) or args.until<=start:
                ap.error("--until must exceed the checkpoint physical time")
            inputs=source.parent/"inputs"
            if source_inputs is not None and local_barth==bool(source_mode) and (inputs/rt.EXE.name).is_file():
                rt.EXE=inputs/rt.EXE.name
            if (inputs/rt.CONFIG.name).is_file():
                rt.CONFIG=inputs/rt.CONFIG.name
            end_label=f"{args.until:g}".replace(".","p")
            folder=study_root/f"continue_from_{step:08d}_to_{end_label}"
            print(f"Continuing {source} from t={start:g} to {args.until:g}",flush=True)
            _,result=run(folder,800,9,256,args.until,restart=source,local_barth=local_barth)
            assert result["step"]>step
            finish(folder,folder/"density_pressure.png")
        else:
            finish(study_root/"full",study_root/"RP1_t05_density_pressure.png")
