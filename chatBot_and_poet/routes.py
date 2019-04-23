import os

from flask import render_template, flash,  request, send_from_directory
from imageai.Detection import ObjectDetection

from __init__ import app
from forms import Write1Form,Write2Form,Write3Form,Write4Form
from poet_py.write_poem import start_model
from transplate.transplate import translate,get_reuslt

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = os.getcwd()
execution_path = os.getcwd()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
writer = start_model()

from chatBot_py import demo_test

size = 8               # LSTM神经元size
GO_ID = 1              # 输出序列起始标记
EOS_ID = 2             # 结尾标记
PAD_ID = 0             # 空值填充0
min_freq = 1           # 样本频率超过这个值才会存入词表
epochs = 2000          # 训练次数
batch_num = 1000       # 参与训练的问答对个数
input_seq_len = 25         # 输入序列长度
output_seq_len = 50        # 输出序列长度
init_learning_rate = 0.5     # 初始学习率
encoder_inputs = ""
decoder_inputs = ""
target_weights = ""
outputs = ""
loss = ""
update = ""
saver = ""
learning_rate_decay_op = ""
learning_rate = ""
chatBot = demo_test.chatBot()
# graph1 = demo_test.tf.Graph()
# sess1 = tf.Session(graph=graph1)
# with graph1.as_default():
# encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate = chatBot.get_model(
#     feed_previous=True)
# saver.restore(demo_test.sess, './output_chat/' + str(epochs) + '/demo_')

# ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# app.config['UPLOAD_FOLDER'] = os.getcwd()
# execution_path = os.getcwd()
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# writer = start_model()

graph1 = demo_test.tf.Graph()
sess1 = demo_test.tf.Session(graph=graph1)
with graph1.as_default():
    encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate = chatBot.get_model(
        feed_previous=True)
    saver.restore(sess1, './output_chat/' + str(epochs) + '/demo_')

# with sess1.as_default():
#     with g1.as_default():
#         tf.global_variables_initializer().run()
#         model_saver = tf.train.Saver(tf.global_variables())
#         model_ckpt = tf.train.get_checkpoint_state(“model1/save/path”)
#         model_saver.restore(sess, model_ckpt.model_checkpoint_path)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/time')
def time():
    return render_template('time.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/write1', methods=['GET', 'POST'])
def write1():
    form = Write1Form()
    if form.validate_on_submit():
        flash(writer.free_verse())
    return render_template('write1.html', title='自由诗', form=form)

@app.route('/write2', methods=['GET', 'POST'])
def write2():
    form = Write2Form()
    if form.validate_on_submit():
        flash(writer.rhyme_verse())
    return render_template('write2.html', title='押韵诗', form=form)

@app.route('/write3', methods=['GET', 'POST'])
def write3():
    form = Write3Form()
    if form.validate_on_submit():
        flash(writer.cangtou(form.headText.data))
    return render_template('write3.html',  title='藏头诗', form=form)

@app.route('/chat', methods=['GET', 'POST'])
def chat():
	return render_template("chat.html")

@app.route('/obtain_answer')
def obtain_answer():
	question = request.args.get('question')
	answer = chatBot.predict(sess1,encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate)
	answer = hearNotUnderstand(answer,question)
	return answer

def hearNotUnderstand(answer,question):
    if "听不懂你在讲什么啦！" in answer:
        answer = answer + "给你写一首诗吧！" + writer.free_verse()
    if "诗" in question:
        if "藏头" in question:
            return "好滴啊 " + writer.cangtou(question.split("，")[1])
        else:
            return "好啦好啦，写就是了  " + writer.free_verse()
    return answer

# def hearcangtou(question):

"""

"""
@app.route('/write4', methods=['GET', 'POST'])
def write4():
    form = Write4Form()
    if form.validate_on_submit():
        flash(writer.hide_words(form.headText.data))
    return render_template('write4.html',  title='藏字诗', form=form)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.template_test('end_with')
def end_with(str, suffix):
    return str.lower().endswith(suffix.lower())

"""
图像成诗：上传一张图片，识别图片内容，根据图片内容作藏头诗
"""
@app.route('/write5', methods=['GET', 'POST'])
def write5():
    filename = ""
    img_name = ""
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = 'static/images/image.jpg'
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            imageAI()
    return render_template('write5.html', title='图像成诗',img_name=filename)

"""
识别接收到的图片内容
"""
def imageAI():
    text=""
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "static/images/image.jpg"), output_image_path=os.path.join(execution_path , "static/images/imagenew.jpg"))
    for eachObject in detections:
        text = text + get_reuslt(translate(eachObject["name"]))
    flash(writer.cangtou(text))


# def predict(self, question):
#     """
#     预测过程
#     """
#     with tf.Session() as sess:
#         encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate = self.get_model(
#             feed_previous=True)
#         saver.restore(sess, './output_chat/' + str(epochs) + '/demo_')
#         # saver.restore(sess, './output_chat/demo')
#         # sys.stdout.write("you ask>> ")
#         # sys.stdout.flush()
#         # input_seq = sys.stdin.readline()
#         # input_seq = chat()
#         input_seq = request.args.get('question')
#         while input_seq:
#             input_seq = input_seq.strip()
#             input_id_list = self.get_id_list_from(input_seq)
#             if (len(input_id_list)):
#                 sample_encoder_inputs, sample_decoder_inputs, sample_target_weights = self.seq_to_encoder(
#                     ' '.join([str(v) for v in input_id_list]))
#
#                 input_feed = {}
#                 for l in range(input_seq_len):
#                     input_feed[encoder_inputs[l].name] = sample_encoder_inputs[l]
#                 for l in range(output_seq_len):
#                     input_feed[decoder_inputs[l].name] = sample_decoder_inputs[l]
#                     input_feed[target_weights[l].name] = sample_target_weights[l]
#                 input_feed[decoder_inputs[output_seq_len].name] = np.zeros([2], dtype=np.int32)
#
#                 # 预测输出
#                 outputs_seq = sess.run(outputs, input_feed)
#                 # 因为输出数据每一个是num_decoder_symbols维的，因此找到数值最大的那个就是预测的id，就是这里的argmax函数的功能
#                 outputs_seq = [int(np.argmax(logit[0], axis=0)) for logit in outputs_seq]
#                 # 如果是结尾符，那么后面的语句就不输出了
#                 if EOS_ID in outputs_seq:
#                     outputs_seq = outputs_seq[:outputs_seq.index(EOS_ID)]
#                 outputs_seq = [self.wordToken.id2word(v) for v in outputs_seq]
#                 # print("chatbot>>", " ".join(outputs_seq))
#                 return " ".join(outputs_seq)
#             else:
#                 # print("WARN：词汇不在服务区")
#                 return "对不起，听不懂你在说什么啊！"
#             # sys.stdout.write("you ask>>")
#             sys.stdout.flush()
#             # input_seq = sys.stdin.readline()
#     return "你倒是说话啊！"

if __name__ == "__main__":
    app.run()