**TỔNG HỢP HƯỚNG NGHIÊN CỨU & KẾ HOẠCH THỰC NGHIỆM**

*Continual Learning cho Vision-Language Models --- hướng MoE-Adapters &
Text-aware Vision Encoder*

Cập nhật: tháng 7/2026

### Mục đích tài liệu

Tài liệu này tổng hợp lại 25 bài báo đã review (17 bài trong file
Literature Review gốc + 8 bài mới bổ sung liên quan trực tiếp đến
MoE-Adapters4CL và hướng text-aware vision encoder), nhóm theo hướng
tiếp cận, chỉ ra kết luận và hạn chế của từng nhóm, phân tích khả năng
của hướng mới (text-conditioned MoE routing), ước lượng overhead/cost,
và đề xuất kế hoạch ngắn hạn.

### Phân nhóm các hướng đã có thực nghiệm --- kết luận và lý do chưa tốt

Các paper hiện có trong continual learning cho VLM có thể chia thành 7
nhóm hướng tiếp cận chính. Dưới đây là tổng hợp kết luận chung và nguyên
nhân các hướng này chưa giải quyết triệt để bài toán.

2.1. Multi-modal Replay (Synthetic / Selective Replay)
------------------------------------------------------

**Paper liên quan:** LoRA-Loop (\#4), VLM-Assisted CVQA cho self-driving
(\#5), GB-VLM (\#7), VLM-C4L (\#9), Neural Sentinel (\#12).

**Kết luận chung:** Sinh dữ liệu replay bằng generator được fine-tune
theo domain (LoRA trên Stable Diffusion, hoặc chính VLM tự sinh) giúp
giảm forgetting tốt hơn generator cố định hoặc replay ngẫu nhiên; kết
hợp confidence-based selection hoặc balancing giúp tránh thiên lệch task
mới.

**Tại sao chưa tốt:**

-   Vẫn phụ thuộc chất lượng generator --- nếu generator không mô tả
    được đặc trưng tinh vi của domain mới, replay sẽ kém hiệu quả (giới
    hạn của LoRA-Loop, GB-VLM).

-   Replay dựa trên dữ liệu tổng hợp hoặc core dataset vẫn cần lưu
    trữ/tính toán thêm (không data-free hoàn toàn --- VLM-C4L, Neural
    Sentinel vẫn giữ buffer).

-   Chi phí huấn luyện cao hơn các hướng chỉ dùng regularization/adapter
    do phải sinh dữ liệu hoặc chạy thêm generator mỗi vòng cập nhật.

-   Chưa đánh giá task-agnostic / cross-domain thực sự (phần lớn vẫn
    biết task-id hoặc domain khi replay).

2.2. Cross-modal Regularization & Semantic/Geometry Alignment
-------------------------------------------------------------

**Paper liên quan:** LGA (\#6), PROOF (\#17), DesCLIP (\#8), SeGP-CL
(\#10), TPPT (\#25), C-CLIP --- Contrastive Knowledge Consolidation
(\#22).

**Kết luận chung:** Dùng text embedding (tên lớp, mô tả thuộc tính, hoặc
prototype) làm \'neo ngữ nghĩa\' ổn định để căn chỉnh đặc trưng ảnh là
hướng hiệu quả, không cần replay buffer, giữ tốt zero-shot. SeGP-CL và
DesCLIP đạt SOTA trên nhiều benchmark exemplar-free.

**Tại sao chưa tốt:**

-   Hiệu quả phụ thuộc mạnh vào chất lượng mô tả/tên lớp --- nếu mô tả
    nghèo hoặc nhiễu (do LLM sinh sai) thì alignment kém đi (hạn chế của
    DesCLIP).

-   Phần lớn dùng text embedding như tín hiệu loss/anchor tĩnh ở tầng
    cuối, chưa can thiệp vào cơ chế lựa chọn tham số/adapter nào được
    kích hoạt --- tức là chưa \'text-aware\' ở cấp routing/kiến trúc,
    chỉ ở cấp biểu diễn.

-   Chủ yếu đánh giá Class-Incremental Learning, chưa mở rộng nhiều sang
    Domain-IL/Task-IL hay VQA/Captioning.

-   PROOF tăng dần số projection theo task → mở rộng kiến trúc theo thời
    gian, gần giống vấn đề của MoE tĩnh.

2.3. Parameter-Efficient Adaptation --- Prompt-based
----------------------------------------------------

**Paper liên quan:** TICL-VLM (\#3), IAP (\#14), TPPT (\#25).

**Kết luận chung:** Prompt theo task hoặc theo từng instance
(Instance-Aware Prompting) giúp CLIP thích nghi mà không cần fine-tune
backbone; kết hợp Prompt Memory Bank hoặc textual anchor giúp giảm xung
đột giữa các task.

**Tại sao chưa tốt:**

-   Mỗi task cần thêm module riêng (feature selection + classifier ở
    TICL-VLM) → số tham số vẫn tăng theo số task, chỉ chậm hơn full
    fine-tuning.

-   Prompt Memory Bank tăng bộ nhớ khi số task lớn; chi phí suy luận
    tăng nhẹ do phải sinh/truy xuất prompt cho từng mẫu (IAP).

-   Chưa kiểm chứng ở quy mô Multimodal LLM (LLaVA, Qwen-VL) --- chỉ
    dừng ở CLIP classification.

2.4. Parameter-Efficient Adaptation --- MoE / Adapter-based (dòng chính của MoE-Adapters4CL)
--------------------------------------------------------------------------------------------

**Paper liên quan:** MoE-Adapters4CL (gốc, CVPR\'24), MoE-Adapters++
(\#18), DIMoE-Adapters (\#19), On Token\'s Dilemma / LLaVA-DyMoE (\#20),
LLaVA-CMoE (\#21), TRGE --- Two-Level Routing (\#23), PASs-MoE (\#24).

**Kết luận chung:** Đây là dòng đang phát triển nhanh nhất (ít nhất 7
biến thể tính đến 07/2026). Xu hướng chung: (a) chuyển từ expert pool
tĩnh sang động (MoE-Adapters++, DIMoE-Adapters), (b) phát hiện và xử lý
lỗi routing ở cấp độ tinh hơn --- token-level (Token\'s Dilemma) hoặc
pathway-level (PASs-MoE) thay vì chỉ task-level, (c) phân nhóm expert
theo domain/task để tách trách nhiệm rõ hơn (TRGE).

**Tại sao chưa tốt / còn khoảng trống:**

-   Tất cả các router hiện có (kể cả các bản 2026 mới nhất) đều quyết
    định chọn expert dựa trên tín hiệu THỊ GIÁC hoặc latent nội bộ (ảnh,
    token embedding, activation pathway) --- CHƯA có công trình nào dùng
    text/class-description embedding làm tín hiệu điều khiển routing một
    cách tường minh. Đây là khoảng trống rõ nhất.

-   Router dựa thuần trên ảnh dễ bị nhiễu domain: ảnh mới lạ có thể
    khiến router chọn sai expert (nguyên nhân gốc của \'routing-drift\'
    mà Token\'s Dilemma chỉ ra), trong khi text mô tả lớp/task thường ổn
    định hơn qua các domain.

-   Càng nhiều biến thể (dynamic expert, token-level, pathway-level) thì
    kiến trúc càng phức tạp, khó huấn luyện ổn định và khó diễn giải ---
    đây cũng là lý do các paper sau thường phải thêm một cơ chế \'ổn
    định hoá\' mới (LEAS, SCEE, PAS).

-   Phần lớn vẫn cần task-id hoặc ít nhất domain-hint ở một bước nào đó
    (TRGE dùng task identifier cho inter-group routing) --- chưa thực sự
    task-agnostic như X-TAIL đòi hỏi.

2.5. Model Fusion
-----------------

**Paper liên quan:** MFCL (\#11).

**Kết luận chung:** Hợp nhất trọng số/biểu diễn giữa mô hình cũ và mới
(thay vì chỉ fine-tune tiếp) cân bằng tốt hơn giữa stability và
plasticity, tương thích với LoRA/Adapter.

**Tại sao chưa tốt:** Chi phí tính toán tăng do phải lưu và hợp nhất
nhiều phiên bản mô hình; fusion sai chiến lược có thể gây negative
transfer; chưa đánh giá trên MLLM hay bài toán sinh.

2.6. Pruning / Compression cho Continual Learning
-------------------------------------------------

**Paper liên quan:** ContinualPrune-VLM (\#13).

**Kết luận chung:** Kết hợp structured pruning động với continual
learning cho Long-Video VLM, có cả forgetting bounds lý thuyết, giảm
đáng kể chi phí tham số/suy luận.

**Tại sao chưa tốt:** Chỉ áp dụng cho Long-Video VLM; tính toán
importance score để pruning tốn thêm chi phí huấn luyện; pruning quá
mạnh vẫn giảm hiệu năng.

2.7. Analytic / Training-free & Đánh giá-Benchmark
--------------------------------------------------

**Paper liên quan:** RAIL (\#1) --- analytic learning không cần
backprop; Survey VLM-CL (\#2); Re-evaluating CVQA (\#16).

**Kết luận chung:** RAIL cho thấy hướng training-free (ridge regression
adapter) có thể đạt SOTA trên X-TAIL mà không cần replay/reference
dataset --- gợi ý rằng bài toán routing/adapter không nhất thiết phải
học bằng gradient descent. Hai paper còn lại chỉ ra chính cộng đồng đang
thiếu benchmark/giao thức đánh giá công bằng, nhiều kết quả SOTA cũ bị
\'ảo\' do thiết lập thực nghiệm ưu ái.

**Hàm ý:** Khi thiết kế hướng text-aware routing mới, nên dùng X-TAIL
(RAIL) hoặc giao thức chuẩn hoá (Re-evaluating CVQA) làm benchmark để
kết quả có sức thuyết phục, tránh lặp lại sai lầm \'so sánh không công
bằng\'.

3. Hướng mới: Text-aware / Text-conditioned MoE Routing cho Vision Encoder
==========================================================================

3.1. Ý tưởng kiến trúc
----------------------

Thay vì router trong MoE-Adapters chỉ nhận đặc trưng ảnh (hoặc
token/latent nội bộ) để quyết định kích hoạt expert nào, bổ sung một
nhánh điều kiện hoá router bằng text embedding --- có thể là embedding
của tên lớp ứng viên, mô tả thuộc tính (attribute description kiểu
DesCLIP), hoặc prototype văn bản (kiểu TPPT) --- để router \'biết ngữ
nghĩa\' của input trước khi chọn expert, không chỉ dựa vào bề mặt hình
ảnh.

### Có thể triển khai theo 3 biến thể tăng dần độ phức tạp:

-   Biến thể nhẹ (Late Fusion Routing): tính gating weights từ
    concat(visual feature, text-prototype trung bình của các lớp đã
    biết), gần với cách TRGE dùng task-prototype nhưng thay prototype
    ảnh bằng prototype văn bản.

-   Biến thể trung bình (Cross-Attention Routing): dùng cross-attention
    giữa đặc trưng ảnh và tập text embedding ứng viên (giống Task
    Prompt-based Feature Selection của TICL-VLM) để sinh ra gating
    vector, thay vì MLP router thông thường.

-   Biến thể sâu (Joint Text-Vision Pathway): mở rộng khái niệm Pathway
    Activation Subspace của PASs-MoE sang không gian đa phương thức ---
    pathway được xác định bởi cả ảnh lẫn text, giúp router ổn định hơn
    qua domain shift vì text ít trôi (drift) hơn ảnh.

3.2. Vì sao chưa ai làm hướng này (khoảng trống)
------------------------------------------------

-   Bài toán ban đầu của MoE-Adapters4CL (2024) đặt trọng tâm vào giảm
    tham số/chi phí huấn luyện, không phải vào chất lượng ngữ nghĩa của
    routing signal --- nên các bản kế thừa (MoE-Adapters++,
    DIMoE-Adapters) tiếp tục tối ưu theo trục \'động hoá kiến trúc\'
    (dynamic expert) thay vì trục \'ngữ nghĩa hoá tín hiệu routing\'.

-   Nhóm quan tâm đến text-aware alignment (LGA, DesCLIP, TPPT, PROOF)
    lại xuất phát từ hướng regularization/prompt, không dùng kiến trúc
    MoE --- nên không có động lực đưa text vào router.

-   Hai hướng (MoE routing và text-anchor alignment) phát triển song
    song, gần như không giao nhau trong các paper hiện có --- đây là lý
    do tự nhiên khiến khoảng trống này tồn tại, không phải vì bất khả
    thi.

3.3. Khó khăn kỹ thuật cần lường trước
--------------------------------------

-   Tại thời điểm suy luận task-agnostic (không biết domain/task, giống
    X-TAIL), không có sẵn \'văn bản đúng\' để đưa vào router --- cần
    dùng tập text ứng viên (toàn bộ hoặc một tập con class name đã thấy)
    rồi router phải tự chọn, có thể làm tăng chi phí O(số lớp) mỗi lần
    forward nếu không tối ưu.

-   Router học đồng thời với expert dễ gặp lại hiện tượng \'misaligned
    co-drift\' mà PASs-MoE chỉ ra --- nếu thêm nhánh text vào router,
    cần kiểm soát để text-signal không bị lấn át bởi visual-signal trong
    quá trình huấn luyện (cần ablation cẩn thận).

-   Cần tránh trùng lặp với TPPT/DesCLIP: phải chứng minh được rằng đưa
    text vào ROUTING (chọn tham số) mang lại lợi ích khác biệt so với
    chỉ đưa text vào LOSS (căn chỉnh biểu diễn) --- nếu không, phản biện
    dễ hỏi \'khác gì TPPT + MoE cộng lại\'.

-   Nguy cơ tăng độ phức tạp huấn luyện: thêm nhánh cross-attention hoặc
    pathway đa phương thức có thể làm chậm hội tụ, cần thử nghiệm nhỏ
    trước khi mở rộng full benchmark.

-   Rủi ro cao nhất: hai paper rất mới (On Token\'s Dilemma, PASs-MoE
    --- đều 2026) đang đi rất gần hướng \'routing signal chất lượng cao
    hơn\'; cần đọc kỹ để đảm bảo hướng text-aware không trùng lặp và có
    thể định vị là bổ sung/khác biệt (semantic-level thay vì
    token-level/pathway-level).

4. Overhead / Chi phí ước tính theo hướng
=========================================

  **Hướng**                                            **Tham số thêm**                         **Chi phí huấn luyện**                           **Chi phí suy luận**                           **Cần replay/buffer?**      **Cần task-id lúc test?**
  ---------------------------------------------------- ---------------------------------------- ------------------------------------------------ ---------------------------------------------- --------------------------- -----------------------------------------
  Replay-based (LoRA-Loop, GB-VLM, VLM-C4L)            Trung bình (LoRA cho generator)          Cao (sinh dữ liệu mỗi vòng)                      Thấp                                           Có                          Thường có
  Cross-modal regularization (LGA, DesCLIP, SeGP-CL)   Thấp                                     Trung bình (thêm loss/anchor)                    Không đổi/rất thấp                             Không                       Không (exemplar-free)
  Prompt-based (TICL-VLM, IAP, TPPT)                   Thấp-Trung bình                          Thấp-Trung bình                                  Tăng nhẹ                                       Không                       Tuỳ phương pháp
  MoE-Adapters (gốc + 6 biến thể 2025-2026)            Trung bình-Cao                           Trung bình                                       Trung bình (TopK routing)                      Không (phần lớn)            Một số bước vẫn cần (TRGE)
  Model Fusion (MFCL)                                  Cao (nhiều phiên bản model)              Cao                                              Thấp sau fusion                                Không                       Không
  Pruning (ContinualPrune-VLM)                         Giảm dần theo thời gian                  Trung bình (importance score)                    Thấp                                           Không                       Không
  Analytic/training-free (RAIL)                        Thấp (ridge regression)                  Rất thấp (không backprop)                        Thấp                                           Không                       Không
  Đề xuất: Text-aware MoE Routing                      Trung bình (thêm text path vào router)   Trung bình-Cao lúc đầu (ablation, tune fusion)   Tăng nhẹ-trung bình (có thể cache text emb.)   Không (mục tiêu thiết kế)   Không (mục tiêu: task-agnostic, X-TAIL)

Ghi chú: chi phí suy luận của hướng đề xuất phụ thuộc nhiều vào biến thể
--- Late Fusion Routing gần như không tăng chi phí (chỉ thêm 1 phép
concat + MLP nhỏ), trong khi Cross-Attention Routing hoặc Joint Pathway
sẽ tăng chi phí tính theo số lớp ứng viên nếu không cache trước text
embedding (có thể cache vì text embedding của class name cố định, không
đổi theo ảnh).

5. Kế hoạch ngắn hạn (4-6 tuần tới)
===================================

Tuần 1: Củng cố nền tảng & xác định gap
---------------------------------------

-   Đọc full-text 3 paper gần nhất, rủi ro trùng lặp cao nhất:
    MoE-Adapters++, On Token\'s Dilemma (LLaVA-DyMoE), PASs-MoE --- ghi
    chú chi tiết cơ chế routing của từng bài để so sánh trực tiếp với ý
    tưởng text-aware.

-   Đọc kỹ TPPT và DesCLIP để chuẩn bị câu trả lời cho câu hỏi \'khác gì
    việc chỉ thêm text vào loss\'.

-   Chạy lại (reproduce) baseline MoE-Adapters4CL gốc trên 1-2 dataset
    nhỏ (ví dụ CIFAR-100 hoặc 1 domain trong MTIL) để có con số baseline
    của riêng mình, tránh chỉ dựa vào số trong paper.

Tuần 2: Proof-of-concept nhỏ cho Late Fusion Routing
----------------------------------------------------

-   Cài đặt biến thể đơn giản nhất: gating = MLP(concat(visual\_feat,
    mean\_text\_prototype)) thay cho gating = MLP(visual\_feat) trong
    code MoE-Adapters4CL.

-   Thử nghiệm trên 2-3 task/domain nhỏ để kiểm tra: (a) có train được
    ổn định không, (b) gating có thực sự dùng thông tin text hay bị bỏ
    qua (kiểm tra qua ablation zero-out text input).

-   Ghi lại chi phí thời gian train/infer so với baseline để có số liệu
    overhead thực tế, đối chiếu với bảng ước tính ở mục 4.

Tuần 3: Mở rộng benchmark & so sánh baseline
--------------------------------------------

-   Chạy trên MTIL (Multi-domain Task-Incremental Learning) đầy đủ để so
    sánh với MoE-Adapters4CL, MoE-Adapters++ (nếu có code), DesCLIP,
    PROOF.

-   Nếu code MoE-Adapters++/DIMoE-Adapters/PASs-MoE chưa public, ưu tiên
    so baseline với MoE-Adapters4CL (đã có code) và RAIL/DesCLIP (đã có
    kết quả công bố) trước.

-   Bắt đầu thử nghiệm task-agnostic trên X-TAIL (theo thiết lập của
    RAIL) --- đây là điểm nhấn khác biệt hoá so với các paper MoE khác
    vẫn cần task-id.

Tuần 4: Ablation & phân tích thất bại
-------------------------------------

-   Ablation: bỏ text-branch, đổi text prototype bằng embedding ngẫu
    nhiên, thay Late Fusion bằng Cross-Attention --- xem đóng góp thật
    sự đến từ đâu.

-   Phân tích lỗi routing (giống cách Token\'s Dilemma phân tích cấp
    token): xem router có route sai domain nhiều hơn/ít hơn baseline
    visual-only khi có thêm text hay không.

-   Nếu Late Fusion không đủ khác biệt, cân nhắc nâng cấp lên
    Cross-Attention Routing (biến thể trung bình) trong 2 tuần tiếp
    theo.

Tuần 5-6: Viết draft & định vị đóng góp
---------------------------------------

-   Viết phần Related Work đối sánh rõ với 2 nhánh: MoE-routing
    (MoE-Adapters, ++, DIMoE, Token\'s Dilemma, PASs-MoE, TRGE) và
    text-anchor (LGA, DesCLIP, TPPT, PROOF) --- nêu rõ hướng đề xuất nằm
    ở giao điểm và khác biệt ở chỗ nào.

-   Chốt benchmark chính (đề xuất: MTIL cho so sánh chuẩn + X-TAIL cho
    điểm nhấn task-agnostic) và bảng kết quả sơ bộ.

-   Xác định venue mục tiêu (CVPR/ICCV/NeurIPS workshop hoặc ICLR tuỳ
    deadline) để canh timeline viết bài hoàn chỉnh.

6. Danh sách paper cần đọc kỹ trước tiên (ưu tiên)
==================================================

  **Ưu tiên**   **Paper**                                  **Lý do đọc kỹ**
  ------------- ------------------------------------------ ---------------------------------------------------------------------------------------------------------------------
  1             MoE-Adapters++ (\#18)                      Bản kế thừa trực tiếp của MoE-Adapters4CL, baseline gần nhất cần vượt qua
  2             On Token\'s Dilemma / LLaVA-DyMoE (\#20)   Routing-drift analysis rất gần hướng \'router signal chất lượng cao hơn\'
  3             PASs-MoE (\#24)                            Cùng trục ý tưởng \'ổn định hoá routing\', nguy cơ trùng lặp cao nếu không định vị rõ
  4             TPPT (\#25)                                Gần nhất với \'text làm anchor\', cần phân biệt rõ anchor-in-loss vs signal-in-routing
  5             DesCLIP (\#8)                              Cách sinh & lọc attribute description có thể tái sử dụng làm nguồn text-prototype
  6             TRGE (\#23)                                Prototype-guided inter-group routing --- gần ý tưởng dùng prototype cho routing, nhưng dùng ảnh chứ không phải text