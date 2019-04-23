from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField,  SubmitField,FileField
from wtforms.validators import ValidationError, DataRequired, Email, EqualTo, \
    Length
from wtforms.validators import DataRequired


class Write3Form(FlaskForm):
    headText = StringField('藏头', validators=[DataRequired()])
    about_me = TextAreaField('写诗', validators=[Length(min=0, max=140)])
    submit = SubmitField('提交')

class Write4Form(FlaskForm):
    headText = StringField('藏字', validators=[DataRequired()])
    about_me = TextAreaField('写诗', validators=[Length(min=0, max=140)])
    submit = SubmitField('提交')

class ChatForm(FlaskForm):
    headText = StringField('聊聊天', validators=[DataRequired()])
    about_me = TextAreaField('我说', validators=[Length(min=0, max=140)])
    submit = SubmitField('输入')

class Write1Form(FlaskForm):
    submit = SubmitField('生成自由诗')

class Write2Form(FlaskForm):
    submit = SubmitField('生成押韵诗')

'''class Write5Form(FlaskForm):
    photo = FileField('图片',validators=[DataRequired()])
    submit = SubmitField('图像成诗')'''