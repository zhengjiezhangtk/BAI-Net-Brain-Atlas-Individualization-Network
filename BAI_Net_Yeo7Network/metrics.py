import tensorflow as tf


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_neighbor_loss(preds, neighbors, mask):
    '''punishment of labelling the vertex out of its area neighbors '''
    preds = tf.nn.softmax(preds, axis=1)
    loss = tf.reduce_mean(preds*neighbors, axis=1)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

def multiclass_dice(preds, labels):
    """Dice with masking dice, N: the number of vertexes; C: the number of classes
    convert preds into onehot encoding with maxvalue 1, others 0
    equation is 2* preds[N*C] * label[N*C] / (preds[N*C].sum()+labels[N*C].sum()+0.0001)
    """
    max_preds = tf.argmax(preds, axis=1)
    max_labels = tf.argmax(labels, axis=1)
    intersection = tf.reduce_sum(tf.cast(tf.equal(max_preds, max_labels),tf.float32))
    union = tf.shape(max_preds)[0]+tf.shape(max_labels)[0]
    union = tf.cast(union, dtype=tf.float32)
    eps = tf.constant(0.0001)
    dice = (2. * intersection) / (union + eps)
    return dice
