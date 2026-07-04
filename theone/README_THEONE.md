# Wire-level DTN check with The ONE

Goal: replace the paper's application-layer overhead abstraction (Table 11, rho-fraction of
payload) with empirical Bundle-Protocol overhead/delivery/latency for Epidemic, PRoPHET, and a
single-custody router.

## 1. Get The ONE (Java)
    git clone https://github.com/akeranen/the-one.git
    cd the-one && ./compile.sh          # needs a JDK
Copy `telepathology_settings.txt` into the-one/ and ensure a map exists (the repo ships
`data/roads.wkt`; the settings point at it).

## 2. Run once per router
Edit the router lines at the bottom of the settings (Epidemic / PRoPHET / SprayAndWait nrofCopies=1),
then:
    ./one.sh -b 1 telepathology_settings.txt
Read `reports/telepathology_dtn_MessageStatsReport.txt` for:
  - delivery_prob        (delivery ratio)
  - overhead_ratio       = (relayed - delivered)/delivered  <- the empirical replication overhead
  - latency_avg / latency_med

## 3. What to report in the paper
Replace/augment Table 11 with the three routers' measured delivery_prob and overhead_ratio.
Expected pattern: single-custody ~ lowest overhead, Epidemic highest, PRoPHET in between —
same ordering as our abstraction, but now with wire-level numbers including BP headers.
Send me the three MessageStatsReport files and I will build the empirical table and rewrite
Section 5.4 to report measured overhead instead of modelled rho fractions.

Note: The ONE models the Bundle Protocol layer (custody, replication, TTL, buffers) but is a
network simulator, not a testbed; it is the standard tool for DTN evaluation.
