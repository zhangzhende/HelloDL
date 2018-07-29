import tensorflow as tf
import myProject.tensorflowDemo.VGG16.save_and_restore.global_variable as global_variable
from myProject.tensorflowDemo.VGG16.save_and_restore import lineRegulation_model as models

model=models.LineRegModel()
x_input=model.x_input
y_output=model.y_output
saver =tf.train.Saver()
sess=tf.Session()
saver.restore(sess=sess,save_path=global_variable.save_path)

result=sess.run(y_output,feed_dict={x_input:[1]})
print(result)