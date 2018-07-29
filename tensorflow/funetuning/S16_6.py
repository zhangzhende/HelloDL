import tensorflow as tf


saver=tf.train.import_meta_graph("./model/ckpt.meta")

graph=tf.get_default_graph()

a_val=graph.get_tensor_by_name("var/a_val:0")

input_placeholder=graph.get_tensor_by_name("input_placeholder:0")
labels_placeholder=graph.get_tensor_by_name("result_placeholder:0")
y_output=graph.get_tensor_by_name("output:0")

with tf.Session() as sess:
    saver.restore(sess=sess,save_path="./model/ckpt")
    result=sess.run(y_output,feed_dict={input_placeholder:[1]})
    print(result)
    print(sess.run(a_val))