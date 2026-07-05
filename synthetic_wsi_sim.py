"""synthetic_wsi_sim.py — application-layer simulation of transmitting a synthetic
50 MB whole-slide image fragmented into ~10,000 bundles over the same outage traces
used in the paper. Pure-python (no torch). This is a THEORETICAL application-layer
abstraction, not a wire-level DTN protocol validation (no BP headers / custody bytes)."""
import json, random

def simulate(name, trace, num_bundles, total_kb, copies, requires_continuous, ctrl_frac):
    if requires_continuous:
        run=0; started=False; delivered=0; last=0
        for slot,up in enumerate(trace):
            if up:
                started=True; run+=1
                if run>=num_bundles: delivered=num_bundles; last=slot; break
            elif started:
                delivered=run; last=slot; break   # pipe breaks: only the unbroken prefix got through
        else:
            delivered=run
    else:
        sent=0; last=0
        for slot,up in enumerate(trace):
            if up and sent<num_bundles: sent+=1; last=slot
            if sent>=num_bundles: break
        delivered=sent
    ratio=100.0*delivered/num_bundles
    overhead_kb=total_kb*max(0,copies-1)+total_kb*ctrl_frac
    return {"delivery_pct":round(ratio,1),"completion_slots":last+1 if delivered else None,
            "overhead_kb":round(overhead_kb,1),"delivered_bundles":delivered}

def run_condition(cond, reliability, num_bundles, total_kb, seed=0):
    random.seed(seed)
    # generate enough slots that a store-forward scheme can deliver all bundles
    max_slots=int(num_bundles/max(reliability,0.05))*3+num_bundles
    trace=[random.random()<=reliability for _ in range(max_slots)]
    total_nodes=5
    return {
      "TCP_IP":     simulate("TCP/IP",trace,num_bundles,total_kb,1,True,0.05),
      "Epidemic":   simulate("Epidemic",trace,num_bundles,total_kb,total_nodes,False,0.10),
      "PRoPHET":    simulate("PRoPHET",trace,num_bundles,total_kb,2,False,0.07),
      "Neural_Codec_DTN": simulate("Codec-Aware DTN",trace,num_bundles,total_kb,1,False,0.02),
    }

if __name__=="__main__":
    WSI_MB=50.0; total_kb=WSI_MB*1024; num_bundles=10000  # ~5 KB/bundle
    conds={"Real-Time":0.98,"Delayed":0.70,"Extreme":0.30}
    out={"payload_MB":WSI_MB,"num_bundles":num_bundles,"bundle_kb":round(total_kb/num_bundles,2),
         "note":"application-layer abstraction; overhead = replication + assumed control fraction, not measured BP/custody/LTP bytes",
         "conditions":{}}
    for c,r in conds.items():
        out["conditions"][c]=run_condition(c,r,num_bundles,total_kb)
    json.dump(out,open("results/synthetic_wsi_results.json","w"),indent=2)
    print(json.dumps(out,indent=2))
