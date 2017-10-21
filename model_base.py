""""
Contains the ClassifierModel class. Which contains all the
boilerplate code necessary to Create a tensorlfow graph, and training
operations.
"""
import tensorflow as tf
import tensorflow.contrib.slim.nets
import numpy as np
import os
import shutil
import time
import pickle

from data_processing import maybe_make_pardir, pickle2obj, obj2pickle, str2file, load_batch_of_images
from viz import train_curves, batch2grid, vizseg

__author__ = "Ronny Restrepo"
__copyright__ = "Copyright 2017, Ronny Restrepo"
__credits__ = ["Ronny Restrepo"]
__license__ = "Apache License"
__version__ = "2.0"


# TODO: URGENT:  load_batch_of_images has not been implemented


# ==============================================================================
#                                                                    PRETTY_TIME
# ==============================================================================
def pretty_time(t):
    """ Given a time in seconds, returns a string formatted as "HH:MM:SS" """
    t = int(t)
    H, r = divmod(t, 3600)
    M, S = divmod(r, 60)
    return "{:02n}:{:02n}:{:02n}".format(H,M,S)


# ##############################################################################
#                                                     IMAGE CLASSIFICATION MODEL
# ##############################################################################
class ImageClassificationModel(object):
    """
    Examples:
        # Creating a Model that inherits from this class:

        class MyModel(ImageClassificationModel):
            def __init__(self, name, img_shape, n_channels=3, n_classes=10, dynamic=False, l2=None, best_evals_metric="valid_acc"):
                super().__init__(name=name, img_shape=img_shape, n_channels=n_channels, n_classes=n_classes, dynamic=dynamic, l2=l2, best_evals_metric=best_evals_metric)

            def create_body_ops(self):
                ...
                self.logits = ...
    """
    evals_dict_keys = ["train_acc", "valid_acc", "train_loss", "valid_loss", "global_epoch"]

    # Lists of scopes of weights to include/exclude from main snapshot
    main_include = None # None includes all variables
    main_exclude = None


    def __init__(self,
                 name,
                 img_shape,
                 n_channels=3,
                 n_classes=10,
                 dynamic=False,
                 l2=None,
                 best_evals_metric="valid_acc",
                 pretrained_snapshot=None,
                 pretrained_include=None,
                 pretrained_exclude=None):
        """ Initializes a Classifier Class
            n_classes: (int)
            dynamic: (bool)(default=False)
                     Load the images dynamically?
                     If the data just contains paths to image files, and not
                     the images themselves, then set to True.

            If logits_func is None, then you should create a new class that inherits
            from this one that overides `self.body()`
        """
        # MODEL SETTINGS
        # TODO: Save the best evals metric to evals dict, and use that as the
        #       default to load up if none is passed in argument.
        self.batch_size = 4
        self.best_evals_metric = best_evals_metric
        self.l2 = l2
        self.img_shape = img_shape
        self.img_width, self.img_height = img_shape
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.dynamic = dynamic
        self.global_epoch = 0

        # PRETRAINED MODEL SETTINGS
        self.pretrained_model = False if pretrained_snapshot is None else True
        self.pretrained_snapshot = pretrained_snapshot
        # Lists of scopes of weights to include/exclude from pretrained snapshot
        self.pretrained_include = pretrained_include
        self.pretrained_exclude = pretrained_exclude

        # IMPORTANT FILES
        self.model_dir = os.path.join("models", name)
        self.snapshot_file = os.path.join(self.model_dir, "snapshots", "snapshot.chk")
        self.best_snapshot_file = os.path.join(self.model_dir, "snapshots_best", "snapshot.chk")
        self.evals_file = os.path.join(self.model_dir, "evals.pickle")
        self.best_score_file = os.path.join(self.model_dir, "best_score.txt")
        self.train_status_file = os.path.join(self.model_dir, "train_status.txt")
        self.tensorboard_dir = os.path.join(self.model_dir, "tensorboard")

        # DIRECTORIES TO CREATE
        self.dir_structure = [
            self.model_dir,
            os.path.join(self.model_dir, "snapshots"),
            os.path.join(self.model_dir, "snapshots_best"),
            os.path.join(self.model_dir, "tensorboard"),
            ]
        self.create_directory_structure()

        # EVALS DICTIONARY
        self.initialize_evals_dict(self.evals_dict_keys)
        self.global_epoch = self.evals["global_epoch"]

    def create_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.create_input_ops()
            self.create_body_ops()
            self.create_preds_op()
            self.create_loss_ops()
            self.create_optimization_ops()
            self.create_evaluation_metric_ops()
            self.create_saver_ops()
            self.create_tensorboard_ops()

    def create_graph_from_logits_func(self, logits_func):
        """ Given a logits function with the following API:

                `logits_func(X, Y, n_classes, alpha, dropout, l2, is_training)`
                Returning: `logits`

                NOTE: that the argument names are what is important, not the
                ordering.
                NOTE: Each of the arguments passed to the logits_func is a
                placeholder.

        Then it creates the full graph for the model.
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.create_input_ops()
            self.logits = logits_func(X=self.X, Y=self.Y, n_classes=self.n_classes, alpha=self.alpha, dropout=self.dropout, l2=self.l2_scale, is_training=self.is_training)
            self.create_preds_op()
            self.create_loss_ops()
            self.create_optimization_ops()
            self.create_evaluation_metric_ops()
            self.create_saver_ops()
            self.create_tensorboard_ops()


    def create_input_ops(self):
        # TODO: This handling of L2 is ugly, fix it.
        if self.l2 is None:
            l2_scale = 0.0
        else:
            l2_scale = self.l2

        with tf.variable_scope("inputs"):
            self.X = tf.placeholder(tf.float32, shape=(None, self.img_height, self.img_width, self.n_channels), name="X") # [batch, rows, cols, chanels]
            self.Y = tf.placeholder(tf.int32, shape=[None], name="Y")   # [batch]
            self.alpha = tf.placeholder_with_default(0.001, shape=None, name="alpha")
            self.is_training = tf.placeholder_with_default(False, shape=(), name="is_training")
            self.l2_scale = tf.placeholder_with_default(l2_scale, shape=(), name="l2_scale")
            self.dropout = tf.placeholder_with_default(0.0, shape=None, name="dropout")

    def create_body_ops(self):
        """Override this method in child classes.
           must return pre-activation logits of the output layer

           Ops to make use of:
               self.is_training
               self.X
               self.Y
               self.alpha
               self.dropout
               self.l2_scale
               self.l2
               self.n_classes
        """
        # default body graph. Override this in your inherited class
        with tf.name_scope("preprocess") as scope:
            x = tf.div(self.X, 255, name="rescaled_inputs")

        with tf.contrib.framework.arg_scope(
            [tf.contrib.layers.conv2d, tf.contrib.layers.fully_connected],
            activation_fn=tf.nn.relu,
            normalizer_fn=tf.contrib.layers.batch_norm,
            normalizer_params={"is_training": self.is_training}
            ):
            x = tf.contrib.layers.conv2d(x, num_outputs=8, kernel_size=3, stride=2)
            x = tf.layers.dropout(x, rate=self.dropout)
            x = tf.contrib.layers.conv2d(x, num_outputs=16, kernel_size=3, stride=2)
            x = tf.layers.dropout(x, rate=self.dropout)
            x = tf.contrib.layers.conv2d(x, num_outputs=32, kernel_size=3, stride=2)
            x = tf.layers.dropout(x, rate=self.dropout)
            x = tf.contrib.layers.flatten(x)
            self.logits = tf.contrib.layers.fully_connected(x, num_outputs=self.n_classes, normalizer_fn=None, activation_fn=None, scope="logits")

    def create_preds_op(self):
        # PREDUCTIONS - get a class value for each sample
        with tf.name_scope("preds") as scope:
            self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1), name=scope)
            self.probs = tf.nn.softmax(self.logits,name="probs") # probability distributions

    def create_evaluation_metric_ops(self):
        # EVALUATION METRIC
        with tf.name_scope("evaluation") as scope:
            # Define the evaluation metric and update operations
            self.evaluation, self.update_evaluation_vars = tf.metrics.accuracy(
                labels=tf.reshape(self.Y, [-1]),
                predictions=tf.reshape(self.preds, [-1]),
                name=scope)
            # Isolate metric's running variables & create their initializer/reset op
            evaluation_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=scope)
            self.reset_evaluation_vars = tf.variables_initializer(var_list=evaluation_vars)

    def create_loss_ops(self):
        # LOSS - Sums all losses even Regularization losses automatically
        with tf.variable_scope('loss') as scope:
            unrolled_logits = tf.reshape(self.logits, (-1, self.n_classes))
            unrolled_labels = tf.reshape(self.Y, (-1,))
            tf.losses.sparse_softmax_cross_entropy(labels=unrolled_labels, logits=unrolled_logits, reduction="weighted_sum_by_nonzero_weights")
            self.loss = tf.losses.get_total_loss()

    def create_optimization_ops(self):
        # OPTIMIZATION - Also updates batchnorm operations automatically
        with tf.variable_scope('opt') as scope:
            self.optimizer = tf.train.AdamOptimizer(self.alpha, name="optimizer")
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # allow batchnorm
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.minimize(self.loss, name="train_op")

    def create_tensorboard_ops(self):
        # # TENSORBOARD
        # self.summary_writer = tf.summary.FileWriter(os.path.join(self.model_dir, "tensorboard"), graph=self.graph)
        # self.summary_op = tf.summary.scalar(name="dummy", tensor=4)

        # TENSORBOARD - To visialize the architecture
        with tf.variable_scope('tensorboard') as scope:
            self.summary_writer = tf.summary.FileWriter(self.tensorboard_dir, graph=self.graph)
            self.dummy_summary = tf.summary.scalar(name="dummy", tensor=1)
            #self.summary_op = tf.summary.merge_all()

    def create_saver_ops(self):
        """ Create operations to save/restore model weights """
        if self.pretrained_model:
            self.pretrained_saver_ops()

        with tf.device('/cpu:0'): # prevent more than one thread doing file I/O
            # Main Saver
            self.main_exclude = None
            main_vars = tf.contrib.framework.get_variables_to_restore(exclude=self.main_exclude)
            self.saver = tf.train.Saver(main_vars, name="saver")

    def pretrained_saver_ops(self):
        """ Create operations to save/restore model weights """
        with tf.device('/cpu:0'): # prevent more than one thread doing file I/O
            # PRETRAINED SAVER
            pretrained_vars = tf.contrib.framework.get_variables_to_restore(include=self.pretrained_include, exclude=self.pretrained_exclude)
            self.pretrained_saver = tf.train.Saver(pretrained_vars, name="pretrained_saver")

            # # REMAINDER INITIALIZER - all others not handled by pretrained snapshot
            # remainder_vars = tf.contrib.framework.get_variables_to_restore(exclude=[var.name for var in pretrained_vars])
            # self.remainder_initializer = tf.variables_initializer(var_list=remainder_vars)

    def create_directory_structure(self):
        """ Ensure the necessary directory structure exists for saving this model """
        for dir in self.dir_structure:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def initialize_evals_dict(self, keys):
        """ If evals file exists, load it, otherwise create one from scratch.
            You should specify the keys you want to use in the dict."""
        if os.path.exists(self.evals_file):
            print("Loading previosuly saved evals file from: \n- ", self.evals_file)
            with open(self.evals_file, mode = "rb") as fileObj:
                self.evals = pickle.load(fileObj)
        else:
            self.evals = {key: [] for key in keys}
            self.evals["global_epoch"] = 0

    def save_evals_dict(self):
        """ Save evals dict to a picle file in models root directory """
        with open(self.evals_file, mode="wb") as fileObj:
            self.evals["global_epoch"] = self.global_epoch
            pickle.dump(self.evals, fileObj, protocol=2) #py2.7 & 3.x compatible

    def snapshot_exists(self, snapshot_file):
        """ Check if a snapshot file exists.
            Designed to overcome a bug/limitation of tensroflows function
            for checking if snapshot exists. In the case where even the
            directory does not exist, here it gracefully returns False,
            instead of throwing an error.
        """
        return os.path.exists(os.path.dirname(snapshot_file)) \
            and tf.train.checkpoint_exists(snapshot_file)

    def initialize_vars(self, session, best=False):
        """ Override this if you set up custom savers """
        # Determine if to use best, or latest snapshot
        if best:
            snapshot_file = self.best_snapshot_file
        else:
            snapshot_file = self.snapshot_file

        # Check if this model already has saved snapshots
        # If not:
        #    Initialize weights using random intializer.
        #    check if using pretrained weights.
        #         if so, then initialize from pretrained weights, random
        #         from others
        try:
            # Determine if it can continue training from a previous run,
            # or if it needs to be intialized from the begining.
            if self.snapshot_exists(snapshot_file):
                print("Restoring parameters from saved snapshot")
                print("-", snapshot_file)
                self.saver.restore(session, snapshot_file)
            elif self.pretrained_model:
                snapshot_file = self.pretrained_snapshot
                print("Initializing from Pretrained Weights")
                print("-", snapshot_file)
                session.run(tf.global_variables_initializer())
                assert self.snapshot_exists(snapshot_file),\
                    "The pretrained weights file does not exist: \n- "\
                    + str(snapshot_file)
                self.pretrained_saver.restore(session, snapshot_file)
            else:
                print("Initializing to new parameter values")
                session.run(tf.global_variables_initializer())
        except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError) as e:
            msg = "===================================================\n"\
                  "ERROR IN INITIALIZING VARIABLES FROM SNAPSHOTS FILE\n"\
                  "===================================================\n"\
                  "Something went wrong  in  loading   the  parameters\n"\
                  "from  the snapshot. This  is  most  likely  due  to\n"\
                  "changes  being  made   to  the   model,   but   not\n"\
                  "changing   the   snapshots   file   path.   Loading\n"\
                  "from a  snapshot  requires  that   the   model   is\n"\
                  "still exaclty the same since the last  time it  was\n"\
                  "saved.\n"\
                  "However, it could also be that the path to the\n"\
                  "snapshot file is incorect.\n"\
                  "\n"\
                  "Either:\n"\
                  "- Check the filepath to the snapshot is correct.\n"\
                  "- Use a different snapshots filepath to create\n"\
                    "new snapshots for this model. \n"\
                  "- or, Delete the old snapshots manually from the \n"\
                    "computer.\n"\
                  "Once you have done that, try again.  See the full\n"\
                  "printout and traceback above  if  this  did  not\n"\
                  "resolve the issue.\n"\
                  "===================================================\n"\
                  "SNAPSHOT FILE: \n" + str(snapshot_file)
            raise ValueError(str(e) + "\n\n\n" + msg)

    def save_snapshot_in_session(self, session, file):
        """Given an open session, it saves a snapshot of the weights to file"""
        # Create the directory structure for parent directory of snapshot file
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))
        self.saver.save(session, file)

    def shuffle_train_data(self, data):
        n_samples = len(data["Y_train"])
        permutation = list(np.random.permutation(n_samples))
        data["X_train"] = data["X_train"][permutation]
        data["Y_train"] = data["Y_train"][permutation]
        return data

    def get_batch(self, i, batch_size, X, Y=None):
        """ Get the ith batch from the data."""
        X_batch = X[batch_size*i: batch_size*(i+1)]
        # Handle dynamic loading option
        if self.dynamic:
            X_batch = load_batch_of_images(X_batch, img_shape=self.img_shape)

        # Batch of labels if needed
        if Y is not None:
            Y_batch = Y[batch_size*i: batch_size*(i+1)]
            return X_batch, Y_batch
        else:
            return X_batch

    def update_status_file(self, status):
        str2file(status, file=self.train_status_file)

    def update_evals_dict(self, **kwargs):
        """ Appends a new value to the specified key/s in the evals dictionary
            eg: update_evals_dict(valid_acc=0.95, valid_loss=0.341)
            will append the value 0.95 to the end of self.evals["valid_acc"]
            and 0.341 to the end of self.evals["valid_loss"] """
        for key in kwargs:
            self.evals[key].append(kwargs[key])

    def create_session(self):
        """ Creates and returns a session. Be careful to close it
            Ideally use it as follows:

            with model.create_session() as session:
                # Do something with the session here
                ...
        """
        session = tf.Session(graph=self.graph)
        return session

    def train(self, data, n_epochs, alpha=0.001, dropout=0.0, batch_size=32, print_every=10, l2=None, aug_func=None, viz_every=10):
        """Trains the model, for n_epochs given a dictionary of data"""
        n_samples = len(data["X_train"])               # Num training samples
        n_batches = int(np.ceil(n_samples/batch_size)) # Num batches per epoch
        print("DEBUG - ", "using aug func" if aug_func is not None else "NOT using aug func")
        with tf.Session(graph=self.graph) as sess:
            self.initialize_vars(sess)
            t0 = time.time()

            try:
                self.update_status_file("training")
                # TODO: Use global epoch
                for epoch in range(1, n_epochs+1):
                    self.global_epoch += 1
                    print("="*70, "\nEPOCH {}/{} (GLOBAL_EPOCH: {})        ELAPSED TIME: {}".format(epoch, n_epochs, self.global_epoch, pretty_time(time.time()-t0)),"\n"+("="*70))

                    # Shuffle the data
                    data = self.shuffle_train_data(data)

                    # Iterate through each mini-batch
                    for i in range(n_batches):
                        X_batch, Y_batch = self.get_batch(i, X=data["X_train"], Y=data["Y_train"], batch_size=batch_size)
                        if aug_func is not None:
                            X_batch = aug_func(X_batch)

                        # TRAIN
                        feed_dict = {self.X:X_batch, self.Y:Y_batch, self.alpha:alpha, self.is_training:True, self.dropout: dropout}
                        loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)

                        # Print feedback every so often
                        if print_every is not None and (i+1)%print_every==0:
                            print("{} {: 5d} Batch_loss: {}".format(pretty_time(time.time()-t0), i, loss))

                    # Save parameters after each epoch
                    self.save_snapshot_in_session(sess, self.snapshot_file)

                    # Evaluate on full train and validation sets after each epoch
                    train_acc, train_loss = self.evaluate_in_session(data["X_train"][:1024], data["Y_train"][:1024], sess)
                    valid_acc, valid_loss = self.evaluate_in_session(data["X_valid"], data["Y_valid"], sess)
                    self.update_evals_dict(train_acc=train_acc, train_loss=train_loss, valid_acc=valid_acc, valid_loss=valid_loss)
                    self.save_evals_dict()

                    # If its the best model so far, save best snapshot
                    is_best_so_far = self.evals[self.best_evals_metric][-1] >= max(self.evals[self.best_evals_metric])
                    if is_best_so_far:
                        self.save_snapshot_in_session(sess, self.best_snapshot_file)

                    # Print evaluations (with asterix at end if it is best model so far)
                    s = "TR ACC: {: 3.3f} VA ACC: {: 3.3f} TR LOSS: {: 3.5f} VA LOSS: {: 3.5f} {}\n"
                    print(s.format(train_acc, valid_acc, train_loss, valid_loss, "*" if is_best_so_far else ""))

                    # # TRAIN CURVES
                    train_curves(train=self.evals["train_acc"], valid=self.evals["valid_acc"], saveto=os.path.join(self.model_dir, "accuracy.png"), title="Accuracy over time", ylab="Accuracy", legend_pos="lower right")
                    train_curves(train=self.evals["train_loss"], valid=self.evals["valid_loss"], saveto=os.path.join(self.model_dir, "loss.png"), title="Loss over time", ylab="loss", legend_pos="upper right")

                    # VISUALIZE PREDICTIONS - once every so many epochs
                    # TODO: Add prediction visualizations

                    str2file(str(max(self.evals[self.best_evals_metric])), file=self.best_score_file)
                self.update_status_file("done")
                print("DONE in ", pretty_time(time.time()-t0))

            except KeyboardInterrupt as e:
                print("Keyboard Interupt detected")
                # TODO: Finish up gracefully. Maybe create recovery snapshots of model
                self.update_status_file("interupted")
                raise e
            except:
                self.update_status_file("crashed")
                raise

    def predict(self, X, batch_size=32, best=True, session=None, probs=False, verbose=True):
        """ Make predictions on data `X`. Returns the most likely class id
            for each training sample in `X`. You can optionally return the
            probability distribution for all the classes instead by setting
            `probs=True`

        Args:
            X:              (np array) inputs
            batch_size:     (int)(default=32)
            best:           (bool)(default=True) Use the best saved snapshot?
                            If set to False, it uses the latest snapshot.
            session:        (None or tensroflow session)(default=None)
                            Pass a currently running session if you are already
                            in a session. Else, it starts a new one.
            probs:          (bool)(default=False) If set to `True` it returns
                            the probability distribution of each class instead
                            of the id of the most likely class.
            verbose:        (bool)(default=True) If `True`, it prints out
                            progress.
        """
        if session is None:
            with tf.Session(graph=self.graph) as sess:
                self.initialize_vars(sess, best=best)
                return self.predict_in_session(X, session=sess, batch_size=batch_size, verbose=verbose, probs=probs)
        else:
            return self.predict_in_session(X, session=session, batch_size=batch_size, verbose=verbose, probs=probs)

    def predict_in_session(self, X, session, batch_size=32, probs=False, verbose=True):
        """ Make predictions on data `X` within a currently running session.
            Returns the most likely class id for each training sample in `X`.
            You can optionally return the probability distribution for all
            the classes instead by setting `probs=True`

        Args:
            X:              (np array) inputs
            session:        (tensroflow session) Currently running session.
            batch_size:     (int)(default=32)
            probs:          (bool)(default=False) If set to `True` it returns
                            the probability distribution of each class instead
                            of the id of the most likely class.
            verbose:        (bool)(default=True) If `True`, it prints out
                            progress.
        """
        # Dimensions
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples/batch_size))
        if probs:
            preds = np.zeros([n_samples, self.n_classes], dtype=np.float32)
            op = self.probs
        else:
            preds = np.zeros(n_samples, dtype=np.uint8)
            op = self.preds
        if verbose:
            print("MAKING PREDICTIONS")
            percent_interval=10
            print_every = n_batches/percent_interval
            percent = 0

        # MAKE PREDICTIONS ON MINI BATCHES
        for i in range(n_batches):
            X_batch = self.get_batch(i, batch_size=batch_size, X=X)
            feed_dict = {self.X:X_batch, self.is_training:False}
            batch_preds = session.run(op, feed_dict=feed_dict)
            preds[batch_size*i: batch_size*(i+1)] = batch_preds.squeeze()

            if verbose and (i+1)%print_every == 0:
                percent += percent_interval
                print("- {} %".format(percent))

        return preds

    def evaluate(self, X, Y, batch_size=32, best=False):
        """Given input X, and Labels Y, evaluate the accuracy of the model"""
        with tf.Session(graph=self.graph) as sess:
            self.initialize_vars(sess, best=best)
            return self.evaluate_in_session(X,Y, sess, batch_size=batch_size)

    def evaluate_in_session(self, X, Y, session, batch_size=32):
        """Evaluate the model on some data (does it in batches).
           Returns a tuple (score, avg_loss)
        """
        # Iterate through each mini-batch
        total_loss = 0
        n_samples = len(Y)
        n_batches = int(np.ceil(n_samples/batch_size)) # Num batches needed

        # Reset the running variables for evaluation metric
        session.run(self.reset_evaluation_vars)

        for i in range(n_batches):
            X_batch, Y_batch = self.get_batch(i, batch_size=batch_size, X=X, Y=Y)
            feed_dict = {self.X:X_batch, self.Y:Y_batch, self.is_training:False}

            loss, preds, confusion_mtx = session.run([self.loss, self.preds, self.update_evaluation_vars], feed_dict=feed_dict)
            total_loss += loss

        score = session.run(self.evaluation)
        avg_loss = total_loss/float(n_batches)
        return score, avg_loss


# ==============================================================================
# ==============================================================================


# ==============================================================================
#                                                       GRAPH_FROM_GRAPHDEF_FILE
# ==============================================================================
def graph_from_graphdef_file(graph_file, access_these, remap_input=None):
    """ Given a tensorflow GraphDef (*.pb) file, it loads up the
        graph specified by that file.

        You need to specify which operations or tensors you want
        to get access to directly by passing a list of the
        operation or tensor names you want to get access to.

        You can also replace the original input tensor
        in the graph with your own tensor.

    Args:
        graph_file:   (str) Path to the GraphDef (*.pb) file
        access_these: (list of strings) A list of all the tensor
                      names you wish to extract. The tensor names
                      MUST EXACTLY match tensor names in the graph.
        remap_input: (dict) Swap out the input tensor in the graph
                     with your own tensor object.
                     A dictionary:
                     - Key is a string of the input tensor name within the
                       saved graph you are loading.
                     - Value is the new tensor object you want
                        to use as the new input to the saved graph instead.
                    Eg:
                        {"input:0": MyPlaceholder}

    Returns: (list)
        requested_ops: List of tensorflow operations or tensor objects
                       that were retreived by the names specified in the
                       `access_these` list.

        NOTE: the remapped input tensor is not returned, as it is
              already a tensor you have access to (since you created
              it outside the function)
    """
    with tf.device('/cpu:0'): # Prevent multiple prallel I/O operations
        with tf.gfile.FastGFile(graph_file, 'rb') as file_obj:
            # Load the graph from file
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_obj.read())

    # Extract particular operations/tensors
    requested_ops = tf.import_graph_def(
        graph_def,
        name='',
        return_elements=access_these,
        input_map=remap_input)
    return requested_ops


# ==============================================================================
#                                                             SEGMENTATION MODEL
# ==============================================================================
class SegmentationModel(ImageClassificationModel):
    evals_dict_keys = ["train_iou", "valid_iou", "train_loss", "valid_loss", "global_epoch"]
    def __init__(self,
                name,
                img_shape=[299, 299],
                n_channels=3,
                n_classes=10,
                dynamic=False,
                l2=None,
                best_evals_metric="valid_iou",
                pretrained_snapshot=None,
                pretrained_include=None,
                pretrained_exclude=None):
        """ """
        # PASS THE ARGUMENTS TO THE PARENT CLASS
        kwargs = locals()
        kwargs.pop("self")
        kwargs.pop("__class__")
        super().__init__(**kwargs)

        # SETTINGS SPECIFIC TO SEGMENTATION
        # TODO: Have an option to ignore void class irrespective of
        # number of total classes.
        if n_classes == 1:
            # technically 1 class is actually two classes (A or not A)
            # But IoU will get calculated differently, so set to single
            # class mode.
            self.n_classes = 2
            self.single_class_mode = True
        else:
            self.n_classes = n_classes
            self.single_class_mode = False

    def create_body_ops(self):
        """Override this method in child classes.
           must return pre-activation logits of the output layer

           Ops to make use of:
               self.is_training
               self.X
               self.Y
               self.alpha
               self.dropout
               self.l2_scale
               self.l2
               self.n_classes
        """
        # default body graph. Override this in your inherited class
        with tf.name_scope("preprocess") as scope:
            x = tf.div(self.X, 255, name="rescaled_inputs")

        with tf.contrib.framework.arg_scope(
            [tf.contrib.layers.conv2d, tf.contrib.layers],
            activation_fn=tf.nn.relu,
            normalizer_fn=tf.contrib.layers.batch_norm,
            normalizer_params={"is_training": self.is_training}
            ):
            # DOWNSAMPLING
            x = tf.contrib.layers.conv2d(x, num_outputs=8, kernel_size=3, stride=2)
            x = tf.layers.dropout(x, rate=self.dropout)
            x = tf.contrib.layers.conv2d(x, num_outputs=16, kernel_size=3, stride=2)
            x = tf.layers.dropout(x, rate=self.dropout)
            x = tf.contrib.layers.conv2d(x, num_outputs=32, kernel_size=3, stride=2)
            x = tf.layers.dropout(x, rate=self.dropout)

        relu = tf.nn.relu
        n_classes = self.n_classes
        conv2d = tf.contrib.layers.conv2d
        deconv = tf.contrib.layers.conv2d_transpose

        # DOWNSAMPLING
        with tf.contrib.framework.arg_scope(\
            [tf.contrib.layers.conv2d],
            padding = "SAME",
            stride = 2,
            activation_fn =tf.nn.relu,
            normalizer_fn=tf.contrib.layers.batch_norm,
            normalizer_params = {"is_training": self.is_training},
            weights_initializer =tf.contrib.layers.xavier_initializer(),
            weights_regularizer =None,
            variables_collections =None,
            trainable =True):

            d1 = conv2d(x, num_outputs=8, kernel_size=3, scope="d1")
            d2 = conv2d(d1, num_outputs=32, kernel_size=3, scope="d2")
            d3 = conv2d(d2, num_outputs=64, kernel_size=3, scope="d3")
            d4 = conv2d(d3, num_outputs=64, kernel_size=3, scope="d4")

        # UPSAMPLING
        with tf.contrib.framework.arg_scope([deconv, conv2d], \
            padding = "SAME",
            stride = 2,
            activation_fn = None,
            normalizer_fn = None,
            normalizer_params = {"is_training": self.is_training},
            weights_initializer = tf.contrib.layers.xavier_initializer(),
            weights_regularizer = None,
            variables_collections = None,
            trainable = True):

            with tf.variable_scope('u3') as scope:
                previous, residual = d4, d3
                u = deconv(previous, num_outputs=n_classes, kernel_size=4, stride=2)
                s = conv2d(residual, num_outputs=n_classes, kernel_size=1, stride=1, activation_fn=relu, scope="skip")
                u3 = tf.add(u, s, name="up")

            with tf.variable_scope('u2') as scope:
                previous, residual = u3, d2
                u = deconv(previous, num_outputs=n_classes, kernel_size=4, stride=2)
                s = conv2d(residual, num_outputs=n_classes, kernel_size=1, stride=1, activation_fn=relu, scope="skip")
                u2 = tf.add(u, s, name="up")

            with tf.variable_scope('u1') as scope:
                previous, residual = u2, d1
                u = deconv(previous, num_outputs=n_classes, kernel_size=4, stride=2)
                s = conv2d(residual, num_outputs=n_classes, kernel_size=1, stride=1, activation_fn=relu, scope="skip")
                u1 = tf.add(u, s, name="up")

            self.logits = deconv(u1, num_outputs=n_classes, kernel_size=4, stride=2, activation_fn=None, scope="logits")

    def create_evaluation_metric_ops(self):
        # EVALUATION METRIC - IoU
        with tf.name_scope("evaluation") as scope:
            # Define the evaluation metric and update operations
            self.evaluation, self.update_evaluation_vars = tf.metrics.mean_iou(
                tf.reshape(self.Y, [-1]),
                tf.reshape(self.preds, [-1]),
                num_classes=self.n_classes,
                name=scope)
            # Isolate metric's running variables & create their initializer/reset op
            evaluation_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=scope)
            self.reset_evaluation_vars = tf.variables_initializer(var_list=evaluation_vars)

    def train(self, data, n_epochs, alpha=0.001, dropout=0.0, batch_size=32, print_every=10, l2=None, aug_func=None, viz_every=10):
        """Trains the model, for n_epochs given a dictionary of data"""
        # TODO: The only difference between this code and the code in
        # ImageClassification.train() is some mentions to IOU. Find a more
        # generic way to recycle the same code.
        n_samples = len(data["X_train"])               # Num training samples
        n_batches = int(np.ceil(n_samples/batch_size)) # Num batches per epoch
        print("DEBUG - ", "using aug func" if aug_func is not None else "NOT using aug func")
        with tf.Session(graph=self.graph) as sess:
            self.initialize_vars(sess)
            t0 = time.time()

            try:
                self.update_status_file("training")
                for epoch in range(1, n_epochs+1):
                    self.global_epoch += 1
                    print("="*70, "\nEPOCH {}/{} (GLOBAL_EPOCH: {})        ELAPSED TIME: {}".format(epoch, n_epochs, self.global_epoch, pretty_time(time.time()-t0)),"\n"+("="*70))

                    # Shuffle the data
                    data = self.shuffle_train_data(data)

                    # Iterate through each mini-batch
                    for i in range(n_batches):
                        X_batch, Y_batch = self.get_batch(i, X=data["X_train"], Y=data["Y_train"], batch_size=batch_size)
                        if aug_func is not None:
                            X_batch, Y_batch = aug_func(X_batch, Y_batch)

                        # TRAIN
                        feed_dict = {self.X:X_batch, self.Y:Y_batch, self.alpha:alpha, self.is_training:True, self.dropout: dropout}
                        loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)

                        # Print feedback every so often
                        if print_every is not None and (i+1)%print_every==0:
                            print("{} {: 5d} Batch_loss: {}".format(pretty_time(time.time()-t0), i, loss))

                    # Save parameters after each epoch
                    self.save_snapshot_in_session(sess, self.snapshot_file)

                    # Evaluate on full train and validation sets after each epoch
                    train_iou, train_loss = self.evaluate_in_session(data["X_train"][:1000], data["Y_train"][:1000], sess)
                    valid_iou, valid_loss = self.evaluate_in_session(data["X_valid"], data["Y_valid"], sess)
                    self.update_evals_dict(train_iou=train_iou, train_loss=train_loss, valid_iou=valid_iou, valid_loss=valid_loss)
                    self.save_evals_dict()

                    # If its the best model so far, save best snapshot
                    is_best_so_far = self.evals[self.best_evals_metric][-1] >= max(self.evals[self.best_evals_metric])
                    if is_best_so_far:
                        self.save_snapshot_in_session(sess, self.best_snapshot_file)

                    # Print evaluations (with asterix at end if it is best model so far)
                    s = "TR IOU: {: 3.3f} VA IOU: {: 3.3f} TR LOSS: {: 3.5f} VA LOSS: {: 3.5f} {}\n"
                    print(s.format(train_iou, valid_iou, train_loss, valid_loss, "*" if is_best_so_far else ""))

                    # # TRAIN CURVES
                    train_curves(train=self.evals["train_iou"], valid=self.evals["valid_iou"], saveto=os.path.join(self.model_dir, "iou.png"), title="IoU over time", ylab="IoU", legend_pos="lower right")
                    train_curves(train=self.evals["train_loss"], valid=self.evals["valid_loss"], saveto=os.path.join(self.model_dir, "loss.png"), title="Loss over time", ylab="loss", legend_pos="upper right")

                    # VISUALIZE PREDICTIONS - once every so many epochs
                    if self.global_epoch%viz_every==0:
                        self.visualise_semgmentations(data=data, session=sess)

                    str2file(str(max(self.evals[self.best_evals_metric])), file=self.best_score_file)
                self.update_status_file("done")
                print("DONE in ", pretty_time(time.time()-t0))

            except KeyboardInterrupt as e:
                print("Keyboard Interupt detected")
                # TODO: Finish up gracefully. Maybe create recovery snapshots of model
                self.update_status_file("interupted")
                raise e
            except:
                self.update_status_file("crashed")
                raise
