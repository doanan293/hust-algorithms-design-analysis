# Literature Review and Strong-Baseline Selection

**Date:** 2026-09-11
**Spec:** `docs/superpowers/specs/2026-09-11-phase2-core-design.md` Section 13
**Outputs:** comparison table `docs/report/data/related_work.csv`, screening record `docs/literature/screening.csv` (every returned record with its query, identifier, decision, and reason), bibliography entries in `docs/report/references.bib`

## 1. Protocol

### 1.1 Sources and queries

The six topics of spec Section 13 were searched on 2026-09-11 with fixed queries on Crossref (`query.bibliographic`, journal articles published from 2016, 15 rows each) and arXiv (conjunctive all-field terms with `UAV OR drone`, 15 results each). Every returned record is listed in `screening.csv`.

| Id | Topic | Query |
|---|---|---|
| Q1 | Deadline-aware UAV relaying | `deadline UAV relay trajectory` |
| Q2 | Relay selection with backhaul | `UAV relay selection backhaul` |
| Q3 | Time-expanded network flow | `time-expanded network flow UAV` |
| Q4 | Stochastic timely delivery | `UAV data collection deadline stochastic channel` |
| Q5 | Joint trajectory and bandwidth design | `joint UAV trajectory bandwidth allocation relay` |
| Q6 | Decomposition with repair | `UAV trajectory decomposition repair heuristic delay` |

A general web search engine was queried with:

| Id | Query |
|---|---|
| W1 | deadline-aware UAV relay trajectory optimization emergency communication timely delivery |
| W2 | UAV relay selection ground relay backhaul capacity joint trajectory disaster network |
| W3 | UAV data collection time-constrained IoT devices deadlines trajectory maximize served devices |
| W4 | joint UAV trajectory and bandwidth allocation relay network block coordinate descent |
| W5 | time-expanded graph UAV data ferrying store-carry-forward delay tolerant network flow |
| W6 | UAV-assisted network probabilistic line-of-sight robust trajectory stochastic channel deadline constrained |
| W7 | backhaul-aware UAV drone base station placement 5G disaster recovery ground relay |
| W8 | UAV mobile relaying store-and-forward information causality delay constraint throughput |
| W9 | UAV-assisted post-disaster emergency network connectivity restoration multi-hop ground nodes trajectory |
| W10 | UAV data collection hard deadline packets scheduling trajectory "deadline" wireless sensor network IEEE |

### 1.2 Screening and criteria

Candidates were screened first by title and then by abstract. The inclusion criterion of the spec — UAV relaying or data collection with time constraints, or relay selection with backhaul — was applied with these definitions:

- **Time constraint:** the abstract states per-demand timing: a deadline, a delay or Age-of-Information limit, delay as an objective, a message time-to-live, or a count of demands served within a stated period. Minimizing mission or flight time, or maximizing throughput over a horizon under buffer causality, does not qualify.
- **Relay selection with backhaul:** the abstract states a choice among ground stations or relays, or an explicit backhaul capacity for aerial nodes.
- **UAV relaying or data collection:** data moves from ground sources through a UAV to a destination, or a UAV collects data from ground nodes. UAV base stations serving users, computation offloading, and reflecting-surface access qualify only through the backhaul criterion.
- The reference paper of the course topic (Huang et al.) is included regardless of the criteria.

### 1.3 Verification

- **Metadata** comes from the Crossref record of the DOI, or from the arXiv record for preprints; every bibliography entry was generated from those records on 2026-09-11.
- **Statements** come only from text that was read: abstracts from arXiv, Semantic Scholar, OpenAlex, or Crossref, and the full text of Huang et al. (`docs/references/sensors-25-07443-v2.md`). A table cell reads *Không nêu* (not stated) when that text is silent.
- A candidate whose abstract could not be retrieved from any of these sources was excluded, because none of its statements could be checked.

### 1.4 Limitations

One reviewer screened all records; extraction is at abstract level except for the reference paper; web search results depend on the engine and date; the conjunctive arXiv queries returned few records.

## 2. Included Papers

Sixteen papers are included. Screening counts: 74 records excluded by title, 18 excluded by abstract, 15 included entries in `screening.csv` (the reference paper is added separately).

| Key | Reference | Criterion | Text read |
|---|---|---|---|
| `huang2025ddatsap` | Huang et al., “Dynamic Dual-Antenna Time-Slot Allocation Protocol for UAV-Aided Relaying System Under Probabilistic LoS-Channel”, *Sensors* 25(24):7443, 2025, DOI 10.3390/s25247443 | reference paper of the course topic | full text (docs/references/sensors-25-07443-v2.md) |
| `tran2022relay` | Tran et al., “UAV Relay-Assisted Emergency Communications in IoT Networks: Resource Allocation and Trajectory Optimization”, *IEEE Transactions on Wireless Communications* 21(3):1621–1637, 2022, DOI 10.1109/twc.2021.3105821 | time: per-device latency; counts devices served on time | arXiv abstract (2008.00218) |
| `samir2020time` | Samir et al., “UAV Trajectory Planning for Data Collection from Time-Constrained IoT Devices”, *IEEE Transactions on Wireless Communications* 19(1):34–46, 2020, DOI 10.1109/twc.2019.2940447 | time: per-device deadline | Semantic Scholar abstract |
| `albusalih2018deadline` | Albu-Salih and Seno, “Optimal UAV Deployment for Data Collection in Deadline-based IoT Applications”, *Baghdad Science Journal* 15(4), 2018, DOI 10.21123/bsj.2018.15.4.0484 | time: collection deadline | Crossref abstract |
| `liu2022aoi` | Liu and Zheng, “UAV Trajectory Optimization for Time-Constrained Data Collection in UAV-Enabled Environmental Monitoring Systems”, *IEEE Internet of Things Journal* 9(23):24300–24314, 2022, DOI 10.1109/jiot.2022.3189214 | time: AoI limit of monitored data | Semantic Scholar abstract |
| `liu2024interference` | Liu and Zheng, “UAV Trajectory Planning With Interference Awareness in UAV-Enabled Time-Constrained Data Collection Systems”, *IEEE Transactions on Vehicular Technology* 73(2):2799–2815, 2024, DOI 10.1109/tvt.2023.3320676 | time: data time constraint; station association | Semantic Scholar abstract |
| `tang2022delay` | Tang et al., “Delay-Tolerant UAV-Assisted Communication: Online Trajectory Design and User Association”, *IEEE Transactions on Vehicular Technology* 71(12):13137–13151, 2022, DOI 10.1109/tvt.2022.3195788 | time: finite queueing delay; user association | Semantic Scholar abstract |
| `he2022delay` | He et al., “Trajectory Optimization and Channel Allocation for Delay Sensitive Secure Transmission in UAV-Relayed VANETs”, *IEEE Transactions on Vehicular Technology* 71(4):4512–4517, 2022, DOI 10.1109/tvt.2022.3144178 | time: delay objective; UAV relaying | Semantic Scholar abstract |
| `cao2023priority` | Cao et al., “Energy-Delay Tradeoff for Dynamic Trajectory Planning in Priority-Oriented UAV-Aided IoT Networks”, *IEEE Transactions on Green Communications and Networking* 7(1):158–170, 2023, DOI 10.1109/tgcn.2022.3196670 | time: priority-weighted delay objective | OpenAlex abstract |
| `fadlullah2016dynamic` | Fadlullah et al., “A dynamic trajectory control algorithm for improving the communication throughput and delay in UAV-aided networks”, *IEEE Network* 30(1):100–105, 2016, DOI 10.1109/mnet.2016.7389838 | time: congestion delay objective; UAV relaying | OpenAlex abstract |
| `du2023timeconstrained` | Du et al., “Time-Constrained UAV-Aided Data Collection for IoT Networks with Energy Harvesting”, *IEEE INFOCOM 2023 - IEEE Conference on Computer Communications Workshops (INFOCOM WKSHPS)*, pp. 1–6, 2023, DOI 10.1109/infocomwkshps57453.2023.10226065 | time: devices served within the collection period | OpenAlex abstract |
| `wang2026juror` | Wang and Yang, “Joint UAV Flight and Opportunistic Routing under Reinforcement Learning for Delay-Tolerant Networks”, *arXiv preprint*, 2026, arXiv:2608.04590 | time: message time-to-live; store-carry-forward | arXiv abstract (2608.04590) |
| `kalantari2017backhaul` | Kalantari et al., “Backhaul-aware robust 3D drone placement in 5G+ wireless networks”, *2017 IEEE International Conference on Communications Workshops (ICC Workshops)*, pp. 109–114, 2017, DOI 10.1109/iccw.2017.7962642 | backhaul capacity | arXiv abstract (1702.08395) |
| `selim2018postdisaster` | Selim and Kamal, “Post-Disaster 4G/5G Network Rehabilitation Using Drones: Solving Battery and Backhaul Issues”, *2018 IEEE Globecom Workshops (GC Wkshps)*, pp. 1–6, 2018, DOI 10.1109/glocomw.2018.8644135 | backhaul capacity | arXiv abstract (1809.07859) |
| `yu2023backhaul` | Yu et al., “Backhaul-Aware Drone Base Station Placement and Resource Management for FSO-Based Drone-Assisted Mobile Networks”, *IEEE Transactions on Network Science and Engineering* 10(3):1659–1668, 2023, DOI 10.1109/tnse.2022.3233004 | backhaul capacity | arXiv abstract (2112.12883) |
| `queiros2024predictive` | Queiros et al., “Joint Channel Bandwidth Assignment and Relay Positioning for Predictive Flying Networks”, *2024 IEEE Globecom Workshops (GC Wkshps)*, pp. 1–6, 2024, DOI 10.1109/gcwkshp64532.2024.11100616 | backhaul relaying | arXiv abstract (2503.14248) |

## 3. Summaries

- **`huang2025ddatsap`.** A UAV decode-and-forward relay with an information buffer serves several two-way user pairs under the PrLoS channel. The DDATSAP protocol shares two antennas within a slot; the resource scheduling factor, transmit power, and trajectory are optimized by BCD with SCA to maximize the minimum average message rate. Section 5.4 gives a per-iteration interior-point complexity of O(R_max (KI)^3) and argues that relaxed auxiliary variables are tight at convergence.
- **`tran2022relay`.** A UAV collects data from latency-constrained IoT devices in an emergency and forwards it to a ground gateway, with limited on-board storage; full-duplex and half-duplex relaying are compared. A device counts as served only if its data reaches the gateway in time. The number of served devices is maximized over bandwidth, power, and trajectory by relaxing binary variables and applying an inner-approximation iterative algorithm; a second problem maximizes throughput for a given number of served devices.
- **`samir2020time`.** A UAV collects data from IoT devices that each have a hard upload deadline. Trajectory and radio resources are optimized to maximize the number of served devices; the mixed-integer non-convex problem is NP-hard. A branch, reduce and bound algorithm gives the global optimum for small instances and an SCA-based algorithm handles larger ones; greedy distance and deadline heuristics are the benchmarks.
- **`albusalih2018deadline`.** Multiple UAVs collect data from IoT nodes within a given deadline while minimizing the energy of nodes and UAVs. The deployment is written as a mixed integer linear program and a heuristic addresses its computational cost.
- **`liu2022aoi`.** A UAV collects time-constrained data in monitoring areas and delivers it to a ground base station. Mission completion time is minimized over speeds, hovering positions, and visiting order subject to Age-of-Information limits and on-board energy, by decomposing into speed (SCA) and path (genetic algorithm) subproblems.
- **`liu2024interference`.** A UAV collects data under data time constraints and delivers it to one of several ground base stations. Mission completion time is minimized over station association, speed, and path with energy and interference constraints, using SCA-based subproblems inside block coordinate descent.
- **`tang2022delay`.** In a delay-tolerant UAV-assisted network with random demand and mobility, UAV trajectories and user association are chosen online to keep users' queueing delay finite while minimizing UAV energy. Lyapunov optimization turns the stochastic problem into per-slot problems, which are decomposed using their graph structure.
- **`he2022delay`.** A UAV relays secure transmissions in a vehicular network. Total information delay is minimized over the relay trajectory (Newton method) and channel allocation (relax-and-round with sequential convex approximation) in an alternating framework.
- **`cao2023priority`.** A UAV collects time-sensitive data from mobile sensor nodes with different delay priorities. A weighted sum of UAV energy and nodes' average delay is minimized through an online reinforcement-learning-based heuristic trajectory planner with attention.
- **`fadlullah2016dynamic`.** Several UAVs form a relay chain for disaster-affected users; the distance between neighbouring trajectory centers drives delay and throughput. UAVs whose queue occupancy exceeds a threshold move their trajectory center and radius; simulations and field experiments evaluate the rule.
- **`du2023timeconstrained`.** Energy-harvesting IoT devices upload data to a UAV by TDMA within a finite gathering period. The number of served devices is maximized over the trajectory, time allocation, and transmit power, using a penalty reformulation with alternating optimization, SCA, and quadratic approximation to reach a sub-optimal solution.
- **`wang2026juror`.** In a delay-tolerant network with intermittent contacts, finite buffers, and message time-to-live, UAV headings and opportunistic routing are learned jointly under centralized training and decentralized execution with proximal policy optimization; the method is compared with PRoPHET and MaxProp.
- **`kalantari2017backhaul`.** The 3D placement of a drone base station is optimized with the capacity of different wireless backhaul types taken into account, maximizing served users (network-centric) or sum rate (user-centric); robustness to user displacement is examined.
- **`selim2018postdisaster`.** A grid of drones restores cellular coverage after a disaster, using tethered drones for high-capacity backhaul and powering drones for charging. Drone energy is minimized over placement while guaranteeing a minimum user rate.
- **`yu2023backhaul`.** A drone base station relays traffic between a macro base station and users over a free-space-optics backhaul whose capacity may be insufficient. Bandwidth allocation and drone placement are optimized jointly under the backhaul capacity constraint by the BROAD algorithm.
- **`queiros2024predictive`.** A second-tier flying relay forwards traffic of first-tier flying nodes whose trajectories are known in advance. Bandwidth assignment and relay position are optimized under positioning limits, finite bandwidth, and minimum rates by simulated annealing with penalty functions.

## 4. Comparison Table

`docs/report/data/related_work.csv` (semicolon-separated, Vietnamese cells for the report) has the columns `key, year, venue, objective, uavs, relay_selection, deadline, connectivity, channel_uncertainty, bandwidth_power, method, guarantee`. Cells follow Sections 2–3; *Không nêu* means the text read does not state it.

None of the included papers combines per-demand deadlines with a choice of ground relays over a finite-capacity backhaul, which is the combination studied in this project; papers with deadlines assume a single destination, and papers with backhaul constraints have no per-demand timing.

## 5. Excluded After Reading the Abstract

| Key or reference | Reason | Use |
|---|---|---|
| `wu2018delay`: Wu and Zhang, “Common Throughput Maximization in UAV-Enabled OFDMA Systems With Delay Consideration”, *IEEE Transactions on Communications* 66(12):6614–6627, 2018, DOI 10.1109/tcomm.2018.2865922 | UAV base station serving users (downlink OFDMA), not relaying or data collection | background: throughput–delay trade-off |
| `park2026expected`: Park and Lee, “Beyond Average-Channel-Based Rate Approximations: UAV Trajectory and Scheduling Optimization With Expected Rate Consideration”, *arXiv preprint*, 2026, arXiv:2602.17019 | communication service to ground nodes; relaying or data collection not stated; mission time objective | background: average-channel rates overestimate expected rates |
| `wu2018multi`: Wu et al., “Joint Trajectory and Communication Design for Multi-UAV Enabled Wireless Networks”, *IEEE Transactions on Wireless Communications* 17(3):2109–2121, 2018, DOI 10.1109/twc.2017.2789293 | UAV base stations maximizing max–min throughput; no time constraint | background: BCD with SCA for UAV trajectories |
| `zeng2016mobile`: Zeng et al., “Throughput Maximization for UAV-Enabled Mobile Relaying Systems”, *IEEE Transactions on Communications* 64(12):4983–4996, 2016, DOI 10.1109/tcomm.2016.2611512 | throughput over a horizon with information causality; no per-demand timing | background: mobile relaying with buffering |
| `na2020emergency`: Na et al., “Joint trajectory and power optimization for UAV-relay-assisted Internet of Things in emergency”, *Physical Communication* 41:101100, 2020, DOI 10.1016/j.phycom.2020.101100 | sum rate over a horizon with information causality; no per-demand timing | background |
| `li2024flight`: Li et al., “Flight time minimization of UAV for cooperative data collection in probabilistic LoS channel”, *China Communications* 21(2):210–226, 2024, DOI 10.23919/jcc.fa.2021-0823.202402 | flight-time minimization; no per-demand timing | background: PrLoS data collection |
| `ropke2006alns`: Ropke and Pisinger, “An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows”, *Transportation Science* 40(4):455–472, 2006, DOI 10.1287/trsc.1050.0135 | not UAV communication | background: large neighbourhood search with repair |
| `jain2004dtn`: Jain et al., “Routing in a delay tolerant network”, *Proceedings of the 2004 conference on Applications, technologies, architectures, and protocols for computer communications*, pp. 145–158, 2004, DOI 10.1145/1015467.1015484 | not UAV communication | background: routing on time-varying graphs with known dynamics |
| Zeng et al., “Trajectory Optimization and Resource Allocation for OFDMA UAV Relay Networks”, IEEE TWC 20(10):6634–6647, 2021, DOI 10.1109/TWC.2021.3075594 | amplify-and-forward relaying for throughput and fairness; no time or backhaul constraint | — |
| Zhou et al., “Delay-Aware UAV Computation Offloading and Communication Assistance for Post-Disaster Rescue”, IEEE TWC 23(12):19110–19125, 2024, DOI 10.1109/TWC.2024.3479709 | delay of task computation at aerial base stations, not delivery of data | — |
| Zhao et al., “UAV-Assisted Emergency Networks in Disasters”, IEEE Wireless Communications 26(1):45–51, 2019, DOI 10.1109/MWC.2018.1800160 | magazine framework; no formulated time or backhaul constraint in the abstract | — |
| Sharifi et al., “UDADT: An Energy- and Delay-Aware Trajectory Planning for UAV-Assisted Wireless Sensor Networks”, Trans. Emerging Telecommunications Technologies 37(6), 2026, DOI 10.1002/ett.70427 | delay is the tour duration; no per-demand timing | — |
| Zhang et al., “Joint Optimization of IRS and UAV-Trajectory ...”, IEEE Vehicular Technology Magazine 17(2):55–63, 2022, DOI 10.1109/MVT.2022.3158047 | IRS-assisted access; no UAV relaying or data collection stated | — |
| Mondal et al., “Deep reinforcement learning based multi-UAV assisted data collection for deadline-sensitive IoT networks”, Physical Communication 73:102880, 2025, DOI 10.1016/j.phycom.2025.102880 | abstract not retrievable (publisher page 403; absent from Crossref, OpenAlex, Semantic Scholar) | — |
| Hu, Chen, Chen, “Joint trajectory-resource optimization for UAV-enabled uplink communication networks with wireless backhaul”, Computer Networks 229:109779, 2023, DOI 10.1016/j.comnet.2023.109779 | abstract not retrievable (same sources) | — |
| Banerjee et al., “EDTP: Energy and Delay Optimized Trajectory Planning for UAV-IoT Environment”, Computer Networks 202:108623, 2022, DOI 10.1016/j.comnet.2021.108623 | abstract not retrievable (same sources) | — |
| “Delay performance of priority-queue equipped UAV-based mobile relay networks: Exploring the impact of trajectories”, Computer Networks 210:108856, 2022, DOI 10.1016/j.comnet.2022.108856 | abstract not retrievable (same sources) | — |
| “A buffer-aware dynamic UAV trajectory design for data collection in resource-constrained IoT frameworks”, Computers and Electrical Engineering 100:107934, 2022, DOI 10.1016/j.compeleceng.2022.107934 | abstract not retrievable (same sources) | — |

Records excluded by title are listed in `screening.csv` with their reason.

## 6. Strong-Baseline Selection

The spec criteria are applied in order. Only three included papers meet the first criterion, an objective close to maximizing the number of alerts delivered on time: `tran2022relay`, `samir2020time`, and `du2023timeconstrained`.

| Criterion | `tran2022relay` | `samir2020time` | `du2023timeconstrained` |
|---|---|---|---|
| 1. Objective close to timely alerts | Served devices whose data reaches the gateway in time | Served devices by their deadlines | Served devices within the gathering period |
| 2. Offline design with known demand | Yes | Yes | Yes |
| 3. Joint trajectory and radio resources | Bandwidth, power, trajectory | Radio resources, trajectory | Time allocation, power, trajectory |
| 4. Detail or code to reproduce | Full text on arXiv (2008.00218) | Full algorithms; unofficial MATLAB/CVX implementation under the MIT license (github.com/willyfh/uav-trajectory-planning) | Six-page workshop paper |
| 5. Adaptable without changing the method | Two hops with UAV storage and delivery to a ground gateway, like the source–UAV–entry-node path of this project | Collection at the UAV; no second hop in the abstract | Collection at the UAV with energy harvesting |

**Decision:** Tran et al. (half-duplex variant) is the strong literature baseline, confirming the default of spec Section 13. Criterion 5 decides between Tran and Samir: only Tran models delivery through the UAV buffer to a ground gateway, which maps onto the UAV paths that enter the ground network. Samir et al. remain the fallback if the Tran adaptation proves infeasible, because code is available. The adaptation — ground entry nodes as gateways with the fixed backhaul routes to the center, alerts as devices, and served as timely — is specified and implemented in sub-project D.

