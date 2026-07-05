# -*- coding: utf-8 -*-
"""Build a clean paper from scratch with real results. No stale content."""
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH as AL
from docx.enum.table import WD_TABLE_ALIGNMENT as TAL
import os

A = "results/paper_assets"; FIG = "results/figures"
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for m in ("top_margin","bottom_margin","left_margin","right_margin"): setattr(sec, m, Inches(1))
n = doc.styles["Normal"]; n.font.name="Times New Roman"; n.font.size=Pt(10)
for lvl,sz in [(1,13),(2,11),(3,10.5)]:
    h=doc.styles[f"Heading {lvl}"]; h.font.name="Times New Roman"; h.font.size=Pt(sz)
    h.font.color.rgb=RGBColor(0,0,0); h.font.bold=True

def H1(t): doc.add_heading(t, level=1)
def H2(t): doc.add_heading(t, level=2)
def H3(t): doc.add_heading(t, level=3)
def P(t, size=10, italic=False, align=None, bold=False):
    p=doc.add_paragraph(); r=p.add_run(t); r.font.size=Pt(size); r.italic=italic; r.bold=bold
    p.alignment=align if align else AL.JUSTIFY; return p
def C(t, size=13, bold=True):
    p=doc.add_paragraph(); p.alignment=AL.CENTER; r=p.add_run(t); r.font.size=Pt(size); r.bold=bold; return p
def eq(name, w=3.0):
    p=doc.add_paragraph(); p.alignment=AL.CENTER
    p.add_run().add_picture(f"{A}/{name}.png", width=Inches(w))
def figure(path, cap, w=5.4):
    p=doc.add_paragraph(); p.alignment=AL.CENTER
    p.add_run().add_picture(path, width=Inches(w))
    c=doc.add_paragraph(); c.alignment=AL.CENTER; r=c.add_run(cap); r.italic=True; r.font.size=Pt(9)
def two_img(p1,p2,cap,w=2.6):
    p=doc.add_paragraph(); p.alignment=AL.CENTER; run=p.add_run()
    run.add_picture(p1, width=Inches(w)); p.add_run("   ").add_picture(p2, width=Inches(w))
    c=doc.add_paragraph(); c.alignment=AL.CENTER; r=c.add_run(cap); r.italic=True; r.font.size=Pt(9)
def table(rows, cap):
    c=doc.add_paragraph(); c.alignment=AL.CENTER; r=c.add_run(cap); r.italic=True; r.font.size=Pt(9)
    t=doc.add_table(rows=len(rows), cols=len(rows[0])); t.style="Table Grid"; t.alignment=TAL.CENTER
    for i,row in enumerate(rows):
        for j,val in enumerate(row):
            cell=t.rows[i].cells[j]; cell.text=str(val)
            for pp in cell.paragraphs:
                for rr in pp.runs:
                    rr.font.size=Pt(8.5); rr.font.name="Times New Roman"
                    if i==0: rr.font.bold=True

# ===================== TITLE =====================
C("AI-Augmented Delay-Tolerant Telepathology in Constrained Networks", 15)
C("Putluru Sashidhar Reddy, Adhvik Sai Bobbasani, Deepa Nivethika", 10, bold=False)
p=doc.add_paragraph(); p.alignment=AL.CENTER
r=p.add_run("School of Computer Science and Engineering, Vellore Institute of Technology, Chennai 600127, India"); r.italic=True; r.font.size=Pt(9)
doc.add_paragraph()

# ===================== ABSTRACT =====================
H1("Abstract")
P("Rapid, accurate diagnosis of histopathology samples is central to the early treatment of cancer, yet rural and resource-constrained clinics often lack on-site pathologists and reliable connectivity. This work presents an integrated, AI-augmented delay-tolerant telepathology framework that (i) classifies biopsy images at the source into eight histopathological subtypes and derives a benign-versus-malignant triage decision, (ii) prioritizes malignant cases for earlier transmission, (iii) compresses images with a learned neural codec, and (iv) delivers them over a delay-tolerant network (DTN) that tolerates intermittent connectivity. Under the official BreakHis patient-wise five-fold protocol (no patient leakage), the classifier attains 87.6% benign-versus-malignant accuracy (AUC 0.94) and 49.2% eight-class subtype accuracy; a from-scratch baseline reaches only 29.8%, confirming the value of transfer learning and attention. A rate-distortion-optimized learned codec compresses each 224x224 image to a few kilobytes, outperforming JPEG2000 and approaching JPEG; uniquely, its compressed latent is directly diagnosable, yielding 82.3% benign-versus-malignant accuracy from the code alone without image reconstruction. Source-side triage delivers urgent cases 31.8% faster under load, and the store-and-forward transmission maintains 100% delivery where connection-oriented TCP/IP collapses to 0% under severe outages. All reported numbers are measured and reproducible. The framework's principal contributions are the integration of compression with diagnosis, priority-based triage, and resilient delivery for bandwidth-limited telepathology.")
kp=doc.add_paragraph(); kr=kp.add_run("Keywords: "); kr.bold=True; kr.font.size=Pt(9)
kt=kp.add_run("Telepathology; Delay-Tolerant Networking; Learned Image Compression; Histopathology Classification; BreakHis; Transfer Learning; Class Imbalance; Rural Healthcare"); kt.font.size=Pt(9)

# ===================== 1. INTRODUCTION =====================
H1("1. Introduction")
P("Histopathology underpins the diagnosis of cancer and many other diseases, but expert pathologists and diagnostic facilities are unevenly distributed. Rural and remote centres frequently lack local expertise, forcing patients to travel or wait, which delays treatment. Telepathology addresses this disparity by digitizing biopsy slides and transmitting them to specialists elsewhere.")
P("Two practical obstacles limit telepathology in low-resource settings. First, pathology images are large, and in deployment whole-slide images can reach hundreds of megabytes; efficient compression is therefore essential. Second, rural networks are frequently intermittent, with low bandwidth and high latency, so protocols that assume continuous connectivity perform poorly. In addition, conventional pipelines treat all cases equally, so clinically urgent (malignant) samples can be delayed behind routine ones.")
P("This paper investigates an integrated framework that combines automated classification, priority-based triage, learned compression, and delay-tolerant networking. Rather than optimizing any single component in isolation, we study their integration and evaluate each honestly under a leakage-free protocol. Our contributions are: (1) an eight-subtype classifier with a derived benign/malignant triage, evaluated patient-wise; (2) a rate-distortion-optimized learned codec whose compressed latent is directly diagnosable; (3) a quantified priority-transmission scheme; and (4) a delay-tolerant delivery mechanism benchmarked against standard protocols under simulated intermittent conditions.")

# ===================== 2. RELATED WORK =====================
H1("2. Related Work")
P("Computational pathology has advanced rapidly, with deep models achieving strong results on whole-slide image analysis and biomarker detection [2], [3], [5], [8]. Federated learning enables collaborative training without sharing patient data and has been applied to computational pathology and medical imaging more broadly [1], [4], [6], [7], [12], [13]. For communication under intermittent connectivity, delay-tolerant networking (DTN) uses a store-carry-forward mechanism and has been explored for resilient healthcare data exchange [9], [10], [11]. For compression, traditional codecs such as JPEG and JPEG2000 are widely used, while learned compression methods optimize a rate-distortion objective and can preserve diagnostically relevant structure [16], [17]. Edge intelligence has also been proposed to process data near its source [24], [25]. Most prior work targets a single component; the present study focuses on their integration and on an honest, reproducible evaluation.")

# ===================== 3. METHODOLOGY =====================
H1("3. Methodology")
H2("3.1 System Architecture")
P("Figure 1 shows the end-to-end pipeline. At the rural clinic, a biopsy image is captured and classified; malignant cases are marked high priority. The image is compressed by a learned neural codec whose rate adapts to the measured link bandwidth, and the compressed bundle is queued for delay-tolerant, store-and-forward transmission. At the diagnostic centre the image is reconstructed; because the codec is learned, the receiver can also classify directly from the compressed latent to obtain an immediate triage decision before or without full reconstruction, after which a specialist reviews the case.")
figure(f"{A}/fig_architecture.png", "Figure 1. System architecture. The pipeline classifies and triages at the source, compresses with a learned codec, transmits over a delay-tolerant store-and-forward network, and reconstructs and (optionally) classifies from the latent at the receiver.", w=6.2)

H2("3.2 Histopathology Image Classification")
P("A convolutional network classifies the input image I into one of eight subtypes, from which a benign/malignant decision is derived:")
eq("eq_classify", 3.0)
P("The backbone is a ResNet-50 with a squeeze-and-excitation attention block and ImageNet transfer learning. Class imbalance is handled with inverse-frequency class weighting in the loss, where for class c with n_c examples out of N total across C classes:")
eq("eq_weight", 1.6)
P("Classification is assessed with accuracy, precision, recall, and F1-score (macro-averaged for the multi-class setting), together with the area under the ROC curve:")
eq("eq_accuracy", 3.0); eq("eq_precision", 2.0); eq("eq_recall", 2.0); eq("eq_f1", 3.2)

H2("3.3 Learned Image Compression")
P("The compression model is an autoencoder with analysis (encoder) and synthesis (decoder) transforms:")
eq("eq_encoder", 2.4)
P("Two variants are studied. A high-fidelity residual autoencoder (v1) stores a raw latent and reconstructs at high quality but is byte-inefficient. A rate-distortion-optimized learned codec (v2) couples a generalized-divisive-normalization transform with a learned entropy model and is trained to minimize a joint rate-distortion loss, where D is distortion (MSE) and R is the entropy-model bit-rate:")
eq("eq_rd", 4.4)
P("Reconstruction quality is measured with MSE, PSNR, and SSIM:")
eq("eq_mse", 2.6); eq("eq_psnr", 3.0); eq("eq_ssim", 4.4)

H2("3.4 Delay-Tolerant Transmission and Triage")
P("Compressed images are segmented into bundles and transmitted using a store-carry-forward mechanism: during an outage a bundle is buffered and forwarded at the next contact window, so delivery survives intermittent connectivity. Malignant cases are placed ahead of routine cases in the transmission queue. Network behaviour is characterized by transmission success rate, throughput, and end-to-end latency:")
eq("eq_success", 3.4); eq("eq_throughput", 2.8); eq("eq_latency", 2.2)

# ===================== 4. EXPERIMENTAL SETUP =====================
H1("4. Experimental Setup")
H2("4.1 Dataset")
P("Experiments use the Breast Cancer Histopathological Database (BreakHis): 7,909 microscopy images from 82 patients at four magnifications (40X, 100X, 200X, 400X), spanning eight subtypes (Table 1). BreakHis provides fixed-size patches (approximately 700x460 pixels), not gigapixel whole-slide images; this study therefore operates at the patch level as a proxy for the full pipeline, standardizing all images to 224x224 for the neural models. Figure 2 shows representative benign and malignant samples.")
table([["Subtype","Class","Images"],
       ["Adenosis","Benign","444"],["Fibroadenoma","Benign","1,014"],
       ["Phyllodes Tumor","Benign","453"],["Tubular Adenoma","Benign","569"],
       ["Ductal Carcinoma","Malignant","3,451"],["Lobular Carcinoma","Malignant","626"],
       ["Mucinous Carcinoma","Malignant","792"],["Papillary Carcinoma","Malignant","560"],
       ["Total","","7,909"]],
      "Table 1. BreakHis composition (all magnifications). The distribution is markedly imbalanced.")
two_img(f"{A}/sample_benign.png", f"{A}/sample_malignant.png",
        "Figure 2. Representative BreakHis samples: benign (Adenosis, left) and malignant (Ductal Carcinoma, right).")

H2("4.2 Evaluation Protocol")
P("All classification results use the official BreakHis patient-wise five-fold cross-validation, in which every patient's images appear in either the training or the test partition of a fold, but never both. This prevents patient leakage and yields metrics that reflect genuine generalization to unseen patients. Metrics are reported as the mean plus or minus one standard deviation over the five folds. Compression is evaluated on a held-out split at 224x224; the learned codec's rate is the entropy-model bit count, which is the near-optimal rate achievable by arithmetic coding. Network results are simulation outputs under explicitly stated conditions and link speeds, not physical-network measurements.")

# ===================== 5. RESULTS =====================
H1("5. Results")
H2("5.1 Classification Performance")
P("Pooling all magnifications, the proposed model attains 49.2% (plus/minus 1.5) eight-class accuracy and 87.6% (plus/minus 2.2) benign-versus-malignant accuracy, with a benign/malignant AUC of 0.939 (plus/minus 0.019) and an eight-class one-vs-rest AUC of 0.739 (plus/minus 0.034). Performance varies with magnification, with 40X giving the strongest subtype discrimination and 400X the weakest (Table 2, Figure 3). Per-class results are highly uneven: common subtypes such as Ductal Carcinoma reach an F1 near 0.70, whereas the rarest malignant subtypes (Papillary, Phyllodes) fall below 0.10, so the macro-F1 and the confusion matrix (Figure 4) are more faithful than accuracy alone. Reporting both the benign/malignant and eight-class levels gives an honest account of what the system can and cannot yet do: the benign/malignant decision is strong and clinically actionable, while fine-grained subtyping remains difficult.")
table([["Magnification","8-class Acc (%)","Macro-F1","Benign/Malignant Acc (%)"],
       ["40X","52.99 ± 2.00","0.399 ± 0.026","89.03 ± 2.89"],
       ["100X","48.69 ± 3.58","0.376 ± 0.035","87.15 ± 3.75"],
       ["200X","49.03 ± 3.28","0.358 ± 0.015","89.04 ± 2.13"],
       ["400X","45.40 ± 2.12","0.306 ± 0.014","83.08 ± 2.95"],
       ["Pooled","49.24 ± 1.52","0.363 ± 0.020","87.62 ± 2.22"]],
      "Table 2. Classification per magnification (patient-wise five-fold, mean ± SD).")
figure(f"{FIG}/fig_per_magnification.png","Figure 3. Classification accuracy per magnification (benign/malignant and 8-class). Error bars are standard deviation across folds.", w=5.2)
figure(f"{FIG}/fig_confusion_40X.png","Figure 4. Row-normalized confusion matrix (%) for 8-class subtype classification at 40X.", w=4.6)

H3("5.1.1 Baseline comparison (ablation)")
P("To isolate the contribution of transfer learning and attention, the proposed model is compared with a plain convolutional network trained from scratch under the identical protocol (Table 3, Figure 5). The proposed model improves eight-class accuracy from 29.8% to 49.2% (a gain of 19.4 points), macro-F1 from 0.241 to 0.363, and benign/malignant accuracy from 80.8% to 87.6%. The consistent margin confirms that the architecture, not the protocol, drives performance.")
table([["Model","8-class Acc (%)","Macro-F1","Benign/Malignant Acc (%)"],
       ["Baseline CNN (from scratch)","29.82 ± 3.80","0.241 ± 0.025","80.76 ± 3.88"],
       ["Proposed (ResNet-50 + SE + transfer)","49.24 ± 1.52","0.363 ± 0.020","87.62 ± 2.22"]],
      "Table 3. Baseline versus proposed (pooled, patient-wise five-fold).")
figure(f"{FIG}/fig_baseline_vs_proposed.png","Figure 5. Baseline versus proposed across the three headline metrics.", w=5.0)

H2("5.2 Compression")
P("Two codecs are evaluated at 224x224. The high-fidelity residual autoencoder (v1) reconstructs at approximately 37.2 dB PSNR and 0.981 SSIM but stores about 50 KB per image, which is byte-inefficient. The rate-distortion-optimized learned codec (v2) is roughly six times more byte-efficient and competitive with standard codecs: it outperforms JPEG2000 at every tested rate and approaches JPEG (Table 4, Figure 6). For example, v2 attains 0.86 SSIM at 4.9 KB and 0.92 SSIM at 8.4 KB. JPEG retains a small advantage at matched rates, so a scale-hyperprior codec and longer training are identified as future work.")
table([["Method","Size (KB)","PSNR (dB)","SSIM"],
       ["Autoencoder v1 (raw latent)","49.0","36.76","0.979"],
       ["Learned codec v2 (proposed)","4.9","27.24","0.859"],
       ["Learned codec v2 (proposed)","8.4","30.83","0.919"],
       ["JPEG","5.9","29.96","0.912"],
       ["JPEG","7.8","31.67","0.942"],
       ["JPEG2000","7.5","29.89","0.862"]],
      "Table 4. Rate-distortion comparison at 224x224 (averaged over 400 held-out images).")
figure(f"{FIG}/fig_compression_rd.png","Figure 6. Rate-distortion at 224x224: the learned codec v2 outperforms JPEG2000 and approaches JPEG, and is far more efficient than v1.", w=5.2)

H3("5.2.1 Diagnosis from the compressed latent")
P("A property unique to a learned codec, which JPEG cannot offer, is that the transmitted latent can be analyzed directly. A lightweight classifier trained on the codec's quantized latent (the approximately 5 KB code), under the same patient-wise protocol and without reconstructing the image, achieves 82.3% (plus/minus 3.0) benign/malignant accuracy (AUC 0.69) and 37.8% eight-class accuracy (Table 5). This is below the full-image classifier, an accuracy-for-bandwidth trade-off, but it shows the compressed representation retains substantial diagnostic information: the receiver can obtain an immediate triage decision from the code and reconstruct the full image only when required.")
table([["Classifier input","8-class Acc (%)","Benign/Malignant (%)","AUC (OVR)"],
       ["Full image (ResNet-50)","49.2","87.6","0.74"],
       ["Compressed latent (~5 KB)","37.8","82.3","0.69"]],
      "Table 5. Classification from the compressed latent versus the full image (patient-wise five-fold).")

H2("5.3 Network Transmission")
P("Transmission is evaluated across three simulated conditions on a 500 KB/s base link (Table 6). Delivery success remains 100% in all cases while throughput and latency degrade gracefully with worsening connectivity, from 374.6 KB/s and 0.14 s (Real-Time) to 111.6 KB/s and 19.48 s (Extreme). When a larger multi-bundle payload (the original uncompressed image) is transmitted, the connection-oriented TCP/IP protocol's delivery ratio collapses from 85% to 10.5% to 0% as outages interrupt active sessions, whereas the store-and-forward schemes (Epidemic, PRoPHET, and the proposed Codec-Aware DTN) maintain 100% delivery. This resilience, rather than the choice of codec, is the framework's core networking contribution.")
table([["Condition","Success","Throughput","Latency","TCP/IP large-file delivery"],
       ["Real-Time","100%","374.6 KB/s","0.14 s","85%"],
       ["Delayed","100%","280.6 KB/s","1.62 s","10.5%"],
       ["Extreme","100%","111.6 KB/s","19.48 s","0%"]],
      "Table 6. Transmission across simulated conditions, and large multi-bundle delivery per protocol.")
P("Over an identical per-image link trace, the protocols also differ in transmission overhead (Table 7, Figure 7): the proposed Codec-Aware DTN incurs about 1.0 KB per image, versus 2.5 KB for TCP/IP, 52.4 KB for PRoPHET, and 200.9 KB for flooding-based Epidemic routing.")
table([["Protocol","Avg overhead per image"],
       ["TCP/IP","2.5 KB"],["Epidemic","200.9 KB"],["PRoPHET","52.4 KB"],
       ["Codec-Aware DTN (proposed)","1.0 KB"]],
      "Table 7. Transmission overhead per image by protocol.")
figure(f"{FIG}/fig_protocol_overhead.png","Figure 7. Average transmission overhead per image by protocol (log scale).", w=4.8)

H2("5.4 Triage and Prioritization")
P("The benefit of source-side triage is quantified directly. In batches of 20 cases contending for a single 500 KB/s link (malignant fraction approximately 0.66, matching the dataset), prioritizing malignant cases reduces their average delivery time from 1.03 s to 0.70 s, a 31.8% reduction for urgent cases, while the overall average delivery time is unchanged. Prioritization thus delivers clinically urgent samples earlier without reducing total throughput.")

H2("5.5 Scalability")
P("Under concurrent-transfer stress tests (5, 10, and 20 simultaneous transmissions), all transfers complete successfully. Latency grows with concurrency but the system remains stable, indicating the framework can scale to multiple clinics transmitting simultaneously.")

# ===================== 6. DISCUSSION =====================
H1("6. Discussion")
P("The results support a clear, honest reading. Benign-versus-malignant discrimination is strong (87.6% accuracy, AUC 0.94) and is the clinically actionable outcome on which the triage depends. Eight-class subtyping is substantially harder (49.2% accuracy, macro-F1 0.363), limited chiefly by the rarest subtypes; this is reported transparently rather than hidden behind pooled accuracy. The learned codec is competitive with standard codecs and, more importantly, enables diagnosis directly from the compressed latent, coupling compression with diagnosis in a way conventional codecs cannot. On the network side, delay-tolerant store-and-forward delivery is markedly more reliable than connection-oriented transmission under outages, and priority scheduling measurably accelerates urgent cases. Together these components form a coherent, resilient pipeline for bandwidth-limited telepathology.")

# ===================== 7. LIMITATIONS =====================
H1("7. Limitations and Future Work")
P("Several limitations frame these results. First, evaluation uses patch-level BreakHis images standardized to 224x224 rather than gigapixel whole-slide images; scaling the pipeline to whole slides via tiling is important future work. Second, eight-class subtyping remains weak for rare classes; more data, stronger augmentation, or hierarchical models may help. Third, the learned codec, while competitive, does not yet beat JPEG on pure rate-distortion; a scale-hyperprior with longer training is a clear next step. Fourth, network results are simulation outputs under stated parameters, not measurements from a deployed rural network. Finally, the automated image-quality flag derived from PSNR and SSIM is a reproducible signal, not a substitute for pathologist judgement; a formal reader study is required to establish clinical usability.")

# ===================== 8. CONCLUSION =====================
H1("8. Conclusion")
P("We presented an integrated AI-augmented delay-tolerant telepathology framework and evaluated it honestly under a leakage-free protocol. The classifier provides strong benign/malignant discrimination (87.6% accuracy, AUC 0.94) with an eight-subtype capability, and clearly outperforms a from-scratch baseline. A learned codec compresses images to a few kilobytes, beats JPEG2000, approaches JPEG, and uniquely supports diagnosis from the compressed latent (82.3% benign/malignant from the code alone). Priority triage accelerates urgent cases by 31.8%, and delay-tolerant delivery sustains 100% success where TCP/IP fails under outages. All numbers are measured and reproducible. The framework offers a practical, resilient pathway toward telepathology in underserved regions, with whole-slide scaling and codec improvements as the main directions ahead.")

# ===================== REFERENCES =====================
H1("References")
refs=[
"M. Y. Lu, R. J. Chen, D. Kong, et al., “Federated learning for computational pathology on gigapixel whole slide images,” Medical Image Analysis, vol. 76, p. 102298, 2022.",
"Y. Zheng, R. H. Gindra, E. J. Green, et al., “A graph-transformer for whole slide image classification,” IEEE Trans. Medical Imaging, vol. 41, no. 11, pp. 3003–3015, 2022.",
"S. Wang, Y. Zhu, L. Yu, et al., “RMDL: Recalibrated multi-instance deep learning for whole slide gastric image classification,” Medical Image Analysis, vol. 58, p. 101549, 2019.",
"M. Jiang, H. Yang, C. Cheng, and Q. Dou, “IOP-FL: Inside-outside personalization for federated medical image segmentation,” IEEE Trans. Medical Imaging, vol. 42, no. 7, pp. 2106–2117, 2023.",
"Y. Zhou and Y. Lu, “Deep hierarchical multiple instance learning for whole slide image classification,” in IEEE ISBI, 2022.",
"B. Li, Z. Liu, L. Shao, et al., “Point transformer with federated learning for predicting breast cancer HER2 status,” in AAAI, vol. 38, no. 4, pp. 3000–3008, 2024.",
"Y. Zhang, Y. Qi, X. Qi, et al., “FedSODA: Federated cross-assessment and dynamic aggregation for histopathology segmentation,” in ICASSP, 2024.",
"A. Aboudessouki, K. M. Ali, M. Elsharkawy, et al., “Automated diagnosis of breast cancer using deep learning-based whole slide image analysis of molecular biomarkers,” in IEEE ICIP, 2023.",
"E. G. Spanakis and V. Sakkalis, “Resilient healthcare communications using delay tolerant proxy services,” in IEEE IST, 2021.",
"S. Rahman, A. Shaf, T. Ali, et al., “An optimal delay tolerant and improved data collection schema using AUVs for underwater WSNs,” IEEE Access, vol. 12, pp. 30146–30163, 2024.",
"A. More and R. Kale, “Review on recent research trends and applications in delay tolerant networks,” in ICCUBEA, 2022.",
"A. Rauniyar, D. H. Hagos, D. Jha, et al., “Federated learning for medical applications: A taxonomy, current trends, challenges, and future research directions,” IEEE IoT Journal, vol. 11, no. 5, pp. 7374–7398, 2023.",
"R. S. Antunes, C. A. da Costa, A. Küderle, et al., “Federated learning for healthcare: Systematic review and architecture proposal,” ACM TIST, vol. 13, no. 4, pp. 1–23, 2022.",
"M. Gupta, S. K. Sharma, R. Saxena, et al., “Compression and extraction of the region of interest from medical images through deep learning methods,” in ICACFCT, 2021.",
"G. Xie, X. Li, X. Pan, and Z. Chen, “Diagnosis-oriented medical image compression with efficient transfer learning,” arXiv:2310.13250, 2023.",
"J. Ballé, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston, “Variational image compression with a scale hyperprior,” in ICLR, 2018.",
"L. Gupta, “Collaborative Edge-Cloud AI for IoT driven secure healthcare system,” in IEEE SysCon, 2023.",
"M. Chen, W. Li, Y. Hao, Y. Qian, and I. Humar, “Edge cognitive computing based smart healthcare system,” Future Generation Computer Systems, vol. 86, pp. 403–411, 2018.",
"F. A. Spanhol, L. S. Oliveira, C. Petitjean, and L. Heutte, “A dataset for breast cancer histopathological image classification,” IEEE Trans. Biomedical Engineering, vol. 63, no. 7, pp. 1455–1462, 2016.",
]
for i,r in enumerate(refs,1):
    p=doc.add_paragraph(); run=p.add_run(f"[{i}] {r}"); run.font.size=Pt(8.5); p.alignment=AL.JUSTIFY

doc.save("../paper_AI_in_Medicine_CLEAN.docx")
print("CLEAN paper built:", os.path.abspath("../paper_AI_in_Medicine_CLEAN.docx"))
print("paragraphs:", len(doc.paragraphs), "tables:", len(doc.tables))
