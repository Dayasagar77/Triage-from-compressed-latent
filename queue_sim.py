# -*- coding: utf-8 -*-
"""
Promote-only priority-queue simulation for latent triage (paper Section 5.5).

Question: under a promote-only scheduler, the latent stage
falsely promotes benign cases (specificity 0.6707 -> 32.93% of benign go to the
fast path) and misses some malignant cases (sensitivity 0.8919 -> 10.81% of
malignant fall to the FIFO low-priority queue). Does the false-positive load on
the fast path cause CATASTROPHIC delay for the missed-malignant cases sitting in
FIFO?

Model: single server (one rural uplink), deterministic service time D per
compressed case (M/D/1), Poisson arrivals rate lambda. Non-preemptive 2-class
priority: HIGH = promoted (true positives + false positives), LOW = not promoted
(true negatives + missed malignant). We measure mean waiting time by class and
compare against a single-FIFO baseline. Cross-checked against Cobham's
non-preemptive priority formula.

ALL PARAMETERS ARE FROM MEASURED RESULT FILES:
  r  (malignant sensitivity)  = latent_confusion.json -> 0.8919
  sp (specificity)            = latent_confusion.json -> 0.6707
  p_mal (malignant fraction)  = triage_results.json   -> 0.664
  D  (service time)           = 49 KB @ 500 kbps       = 0.784 s  (network_results avg_compressed_kb, triage speed)
This is a SIMULATION, reported as such.
"""
import json, random, heapq, statistics, math, os

random.seed(20260704)

R_SENS = 0.8919      # latent malignant sensitivity (latent_confusion.json)
SPEC   = 0.6707      # latent specificity           (latent_confusion.json)
P_MAL  = 0.664       # malignant fraction           (triage_results.json avg_malignant_fraction)
D      = 49.0*8/500.0  # deterministic service time s: 49 KB @ 500 kbps = 0.784 s

def p_promote():
    # P(case routed to fast/high-priority path)
    return P_MAL*R_SENS + (1-P_MAL)*(1-SPEC)

def cobham_nonpreempt(lam, D, frac_high):
    """Cobham non-preemptive M/D/1 2-class priority mean WAITING times.
    Deterministic service: E[S]=D, E[S^2]=D^2. Residual R = lam*E[S^2]/2."""
    rho = lam*D
    if rho >= 1: return None
    lam_h = lam*frac_high; lam_l = lam*(1-frac_high)
    rho_h = lam_h*D; rho_l = lam_l*D
    Res = lam*D*D/2.0
    W_h = Res/(1-rho_h)
    W_l = Res/((1-rho_h)*(1-rho_h-rho_l))
    W_fifo = Res/(1-rho)   # M/D/1 single-queue baseline
    return dict(rho=rho, W_high=W_h, W_low=W_l, W_fifo=W_fifo,
                frac_high=frac_high, rho_high=rho_h, rho_low=rho_l)

def simulate(lam, D, n=400000, warm=20000):
    """Discrete-event non-preemptive 2-class priority M/D/1 Monte Carlo.
    Returns mean waiting time (queue delay, excl. service) per class + FIFO."""
    # generate arrivals
    t=0.0; arr=[]
    for i in range(n):
        t += random.expovariate(lam)
        malignant = random.random() < P_MAL
        if malignant:
            promoted = random.random() < R_SENS
        else:
            promoted = random.random() < (1-SPEC)   # false positive
        arr.append([t, 0 if promoted else 1, malignant, promoted])  # class 0=high
    # ---- non-preemptive priority server ----
    # event-simulate: server free at time 'free'. Maintain two FIFO queues by
    # priority; when server frees, pick highest-priority waiting job.
    # Efficient approach: process arrivals in time order, but service picks by prio.
    import collections
    qh=collections.deque(); ql=collections.deque()
    free=0.0; idx=0; N=len(arr)
    waits_h=[]; waits_l=[]; wm_missed=[]  # missed malignant = low prio & malignant
    # We simulate by stepping the server. Use an event loop over arrivals+departures.
    # Simplify: since single server, replay: next event = min(next arrival, server free)
    # Track queue; when server free and jobs waiting, start highest prio.
    server_busy_until=0.0
    # We iterate a virtual clock.
    pending=[]  # not needed
    # Two-pointer: process arrival stream; when server becomes free, dispatch.
    # Implement proper DES:
    ev=[]  # heap of (time, type) type0=arrival index, type1=departure
    for i,a in enumerate(arr):
        heapq.heappush(ev,(a[0],0,i))
    server_free_at=0.0; busy=False
    def start(job, now, waits_h,waits_l,wm_missed):
        w=now-job[0]
        if job[1]==0: waits_h.append(w)
        else:
            waits_l.append(w)
            if job[2]: wm_missed.append(w)  # malignant that was missed
        return now+D
    while ev:
        time,typ,i = heapq.heappop(ev)
        if typ==0:  # arrival
            job=arr[i]
            if not busy:
                busy=True
                dep=start(job,time,waits_h,waits_l,wm_missed)
                heapq.heappush(ev,(dep,1,-1))
            else:
                if job[1]==0: qh.append(job)
                else: ql.append(job)
        else:  # departure
            if qh:
                job=qh.popleft(); dep=start(job,time,waits_h,waits_l,wm_missed)
                heapq.heappush(ev,(dep,1,-1))
            elif ql:
                job=ql.popleft(); dep=start(job,time,waits_h,waits_l,wm_missed)
                heapq.heappush(ev,(dep,1,-1))
            else:
                busy=False
    # drop warmup
    def m(x): 
        x=x[warm//4:] if len(x)>warm//4 else x
        return statistics.mean(x) if x else 0.0
    return dict(W_high=m(waits_h), W_low=m(waits_l), W_missed_malignant=m(wm_missed),
                n_high=len(waits_h), n_low=len(waits_l), n_missed=len(wm_missed))

def fifo_baseline_sim(lam,D,n=400000,warm=20000):
    t=0.0; arr=[]
    for i in range(n):
        t+=random.expovariate(lam); arr.append(t)
    ev=[(a,) for a in arr]
    free=0.0; waits=[]
    # single FIFO server
    free=0.0
    for a in arr:
        start=max(a,free); waits.append(start-a); free=start+D
    waits=waits[warm:]
    return statistics.mean(waits)

fh=p_promote()
print(f"D={D:.4f}s  P(promote)={fh:.4f}  (p_mal={P_MAL}, r={R_SENS}, spec={SPEC})")
rows=[]
for rho in [0.3,0.5,0.7,0.9]:
    lam=rho/D
    th=cobham_nonpreempt(lam,D,fh)
    sim=simulate(lam,D)
    fifo=fifo_baseline_sim(lam,D)
    row=dict(rho=rho,
             frac_promoted=round(fh,4),
             W_high_theory=round(th['W_high'],4), W_high_sim=round(sim['W_high'],4),
             W_low_theory=round(th['W_low'],4),  W_low_sim=round(sim['W_low'],4),
             W_missed_malignant_sim=round(sim['W_missed_malignant'],4),
             W_fifo_baseline_theory=round(th['W_fifo'],4), W_fifo_baseline_sim=round(fifo,4),
             ratio_missedmal_to_fifo=round(sim['W_missed_malignant']/max(fifo,1e-9),3))
    rows.append(row)
    print(f"rho={rho}: Wlow_th={th['W_low']:.3f} Wlow_sim={sim['W_low']:.3f} "
          f"Wmissed_sim={sim['W_missed_malignant']:.3f} Wfifo={fifo:.3f} "
          f"ratio(missed/fifo)={row['ratio_missedmal_to_fifo']}")

out=dict(
  description="Promote-only non-preemptive M/D/1 priority-queue SIMULATION. "
              "HIGH=promoted (true+false positives), LOW=not promoted (incl. missed malignant).",
  parameters=dict(sensitivity_r=R_SENS, specificity=SPEC, malignant_fraction=P_MAL,
                  service_time_s=round(D,4), frac_promoted=round(fh,4),
                  service="deterministic (M/D/1)", arrivals="Poisson", seed=20260704,
                  sources="latent_confusion.json (r,spec); triage_results.json (p_mal); 49KB@500kbps (D)"),
  rows=rows)
with open("results/queue_sim_results.json","w") as f: json.dump(out,f,indent=2)
print("saved results/queue_sim_results.json")
