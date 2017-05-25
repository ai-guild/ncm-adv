class Trainer(object):

    def __init__(self, num_epochs, batch_size):

        self.num_epochs = 20
        self.B = batch_size

    def fit(self, sess, model, data_src, nmsgs_per_epoch=10):

        B = self.B

        interval = len(idx_q)//B//nmsgs_per_epoch
        for i in range(num_epochs):
            avg_loss = 0.

            for j in range(len(idx_q)//B):

                q_j, a_j = data_src.batch(j)

                _, loss_v = sess.run([model.train_op, model.loss], feed_dict = {
                                inputs  : q_j,
                                targets : a_j
                            })

                avg_loss += loss_v

                if j and j%interval == 0:
                    print('{}.{} : {}'.format(i,j,avg_loss/interval))
                    avg_loss = 0.
