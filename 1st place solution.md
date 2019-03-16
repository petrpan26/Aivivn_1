---


---

<p>Chào mọi người bọn mình là Hoàng và Nhật trong team HoangNhat2 trên leaderboard. Đầu tiên, bọn mình muốn cảm ơn anh Tiệp và mọi người trong team Aivivn vì đã tổ chức một cuộc thi về Machine Learning về xử lý tiếng Việt rất thú vị. Bọn mình đã học được rất nhiều thứ qua cuộc thi.</p>
<h2 id="tóm-tắt-solution">Tóm tắt solution:</h2>
<p>Tụi mình không có nhiều kiến thức về xử lý NLP nên tụi mình tập trung thử nghiệm những model DL và xem model nào hoạt động tốt. Qua những lần thử rất nhiều model tụi mình nhận ra là không có single model nào vượt qua được 0.89x ở Public LB mặc dù có một số làm rất tốt ở local validation. Sau đó, bọn mình đã thôi thử model mới mà đã qua thử nghiệm một số cách kết hợp model hoặc augment train data.</p>
<p>Sau những lần thử nghiệm để đạt được độ diversity phù hợp thì solution được top 1 của bọn mình là Weighted Ensemble của những model sau đây:</p>
<ol>
<li>TextCNN (Weight: 0.1)  <a href="https://richliao.github.io/supervised/classification/2016/11/26/textclassifier-convolutional/">source</a></li>
<li>VDCNN (Weight: 0.1) <a href="https://arxiv.org/abs/1606.01781">source</a></li>
<li>HARNN (Weight: 0.3)</li>
<li>SARNN (Weight: 0.5)</li>
</ol>
<p>Pretrained Embeddings tụi mình test và sử dụng là:</p>
<ul>
<li>word2vecVN (window-size 5, 400dims) <a href="https://github.com/sonvx/word2vecVN">source</a></li>
</ul>
<p>Tụi mình chủ yếu train model ở trên Google Colab và sử dụng GPU của Colab. Thời gian train mỗi model khoảng từ 1-2 tiếng.</p>
<h2 id="chi-tiết-solution">Chi tiết solution:</h2>
<h3 id="models">1.  Models:</h3>
<h4 id="textcnn">1.1 TextCNN</h4>
<h4 id="vdcnn">1.2 VDCNN</h4>
<h4 id="harnn">1.3 HARNN</h4>
<h4 id="sarnn">1.4 SARNN</h4>
<h3 id="ensemble">2. Ensemble:</h3>

