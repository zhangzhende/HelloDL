import tensorflow as tf
import numpy as np
import myProject.tensorflowDemo.VGG16.save_and_restore.global_variable as global_variable
from myProject.tensorflowDemo.VGG16.save_and_restore import lineRegulation_model as models

train_x = np.random.rand(5)
train_y = 5 * train_x + 3.2
model = models.LineRegModel()

a_val = model.a_val
b_val = model.b_val

x_input = model.x_input
y_label = model.y_label

y_output = model.y_output

loss = model.loss
optimizer = model.get_optimizer()
saver = tf.train.Saver()

if __name__ == "__main__":
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    flag = True
    epoch = 0
    while flag:
        epoch += 1
        _, loss_val = sess.run([optimizer, loss], feed_dict={x_input: train_x, y_label: train_y})
        if loss_val < 1e-6:
            flag = False
    print(a_val.eval(session=sess), "     ", b_val.eval(session=sess))
    print("-------%d---------" % epoch)
    saver.save(sess, save_path=global_variable.save_path)
    print("model save finished!")
    sess.close()
