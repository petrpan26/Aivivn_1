---


---

<h1 id="st-place-solution-analysis">1st place solution analysis</h1>
<p>Chào mọi người bọn mình là Hoàng và Nhật trong team HoangNhat2 trên leaderboard. Đầu tiên, bọn mình muốn cảm ơn anh Tiệp và mọi người trong team Aivivn vì đã tổ chức một cuộc thi về Machine Learning về xử lý tiếng Việt rất thú vị. Bọn mình đã học được rất nhiều thứ mới lạ qua cuộc thi.</p>
<h2 id="tóm-tắt-cách-làm">Tóm tắt cách làm:</h2>
<p>Tụi mình không có nhiều kiến thức về xử lý NLP nên tụi mình tập trung thử nghiệm những model DL và xem model nào hoạt động tốt. Qua những lần thử rất nhiều model tụi mình nhận ra là không có single model nào vượt qua được 0.89x ở Public LB mặc dù có một số làm rất tốt ở local validation. Sau đó, bọn mình đã thôi thử model mới mà đã qua thử nghiệm một số cách kết hợp model hoặc augment train data.</p>
<p>Sau những lần thử nghiệm để đạt được độ diversity phù hợp thì solution được top 1 của bọn mình là Weighted Ensemble của những model sau đây:</p>
<ol>
<li>TextCNN (Weight: 0.1)  <a href="https://richliao.github.io/supervised/classification/2016/11/26/textclassifier-convolutional/">source</a></li>
<li>Inspired VDCNN (Weight: 0.1) <a href="https://arxiv.org/abs/1606.01781">source</a></li>
<li>HARNN (Weight: 0.3) <a href="https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf">source</a></li>
<li>SARNN (Weight: 0.5) <a href="https://github.com/CyberZHG/keras-self-attention">source</a></li>
</ol>
<p>Pretrained Embeddings tụi mình test và sử dụng là:</p>
<ul>
<li>word2vecVN (window-size 5, 400dims) <a href="https://github.com/sonvx/word2vecVN">source</a></li>
</ul>
<p>Tụi mình chủ yếu train model ở trên Google Colab và sử dụng GPU của Colab. Thời gian train mỗi model khoảng từ 10 - 20 phút (model converge sau khoảng 5-10 epochs). Những model CNN thì train  nhanh hơn những model RNN rất nhiều vì có thể nó không phải là sequential model nên nó tận dụng được GPU tốt hơn lúc train.</p>
<h2 id="chi-tiết-cách-làm">Chi tiết cách làm:</h2>
<h3 id="models">1.  Models:</h3>
<h4 id="textcnn">1.1 TextCNN:</h4>
<p>Đây là model CNN cho text classification của bọn mình.</p>
<h4 id="architecture">Architecture:</h4>
<p><img src="https://imgur.com/TaEIrPx.png" alt="TextCNN"></p>
<h4 id="vdcnn">1.2 VDCNN:</h4>
<p>Tương tự như TextCNN nhưng ở giữa các layer Convolution có những Residual layer để tránh việc vanishing gradient.</p>
<h4 id="architecture-1">Architecture:</h4>
<p><img src="https://imgur.com/CVPVvd3.png" alt="VDCNN"></p>
<h4 id="harnn">1.3 HARNN:</h4>
<p>HARNN xử lý text ở hai level:</p>
<ol>
<li>Tính encoding cho từng sentence bằng word embedding trong paragraph bằng một BiLSTM</li>
<li>Dùng một BiLSTM để tính document encoding theo sentence encoding.</li>
</ol>
<p>Giữa mỗi layer đều có một Attention layer.</p>
<h4 id="architecture-word2sent">Architecture Word2Sent</h4>
<p><img src="https://imgur.com/JuZzSMM.png" alt="VDCNN"></p>
<h4 id="architecture-sent2doc">Architecture Sent2Doc:</h4>
<p><img src="https://imgur.com/ELAREeE.png" alt="VDCNN"></p>
<h4 id="sarnn">1.4 SARNN:</h4>
<p>Đây là model BiLSTM với Attention ỡ giữa hai layer BiLSTM.</p>
<h4 id="architecture-2">Architecture:</h4>
<p><img src="https://imgur.com/qpF9tPR.png" alt="VDCNN"></p>
<h3 id="combine-models">2. Combine models:</h3>
<p>Bọn mình đã thử những cách kết hợp các models như Stacking and Ensembling nhưng thấy Ensembling đưa ra được kết quả khả quan nhất. Về cách lựa chọn weight thì bọn mình đã dựa vào model nào có kết quả tốt nhất trên Public LB và cho model đó weight cao nhất. Bọn mình để nguyên probability và chọn threshold là 0.5 chứ không tìm threshold vì không thấy được kết quả tăng nhiều.</p>
<h2 id="ngoài-lề">Ngoài lề:</h2>
<h3 id="khó-khăn">Khó khăn:</h3>
<ul>
<li>Có lẽ vấn đề đâu tiên hai đứa gặp phải là vấn đề máy móc. Do hai chiếc máy Macbook 128gb (1 Air, 1 Pro) nên hai đứa không đứa nào đủ chỗ để tải pretrained model về thử mỗi lần tải một cái phải xoá cái cũ đi. Mãi sau này tụi mình chuyển mọi thứ lên Google Colab và Github thì mọi thứ mới bắt đầu nhanh và dễ hơn. Nên bọn mình khuyên các bạn nên xài Google Colab hoặc Kaggle Instance.</li>
<li>Bọn mình gặp nhiều vấn đề với việc reproduce được kết quả với hai lý do. Hàm save_weight của Keras có rất nhiều vấn đề và sau khi load lại thì hầu như model bị hư + trong lúc xử lý có nhiều thứ việc bị undeterministic (Python set, keras model). Model đầu tiên trên 0.9 của tui mình cũng là do chạy lại một model cũ mà thành xD.<br>
<img src="https://cdn-images-1.medium.com/freeze/max/1000/1*wdSexjsOvnksIWNq3MeXhw.jpeg?q=20" alt=""></li>
</ul>
<h3 id="những-cách-tiếp-cận-bọn-mình-đã-thử">Những cách tiếp cận bọn mình đã thử:</h3>
<ul>
<li>Dùng Language Model như Elmo <a href="https://github.com/HIT-SCIR/ELMoForManyLangs">(source)</a>. Approach này có vẻ không phù hợp vì thời gian train quá lâu (thời gian train một epoch bằng thời gian train một model CNN hoặc RNN) và bọn mình cũng không có thời gian để preprocess data lại cho đúng format của Elmo.</li>
<li>Một vấn đề mà bọn mình đã thấy là về việc lượng data không đủ để có thể làm model có thể vượt qua được mức 0.89 - 0.9. Bọn mình đã thử một số cách để augment ra data mới như:</li>
</ul>
<ol>
<li>Thay ngẫu nhiên những từ trong câu bằng từ đồng nghĩa. Bọn mình làm điều này bằng cách thay mỗi từ bằng từ có word embeddings gần nó nhất trong từ điển của bọn mình (Nearest neighbor). Mặc dù thay đổi này không mang lại improvement đáng kể nhưng mình nghĩ với thesaurus tốt hoặc metrics chọn vector phù hợp thì sẽ có thể có kết quả tốt.</li>
<li>Xáo các câu trong HARNN model để có thể generate được nhiều document khác nhau.</li>
<li>Dịch từ tiếng Việt sang các thứ tiếng khác và dịch ngược lại. Mà giờ Google Translate ban cái này rồi ;__;</li>
</ol>
<ul>
<li></li>
</ul>

