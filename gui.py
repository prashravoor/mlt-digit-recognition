import pygubu
import tkinter as tk
import sys
import svm
import cnn
from tkinter import filedialog


class StdoutRedirector():
    def __init__(self, text_area, master):
        self.text_area = text_area
        self.master = master

    def write(self, line):
        self.text_area.insert('end', line)
        self.text_area.see('end')

    def flush(self):
        self.master.update_idletasks()


class Application(pygubu.TkApplication):
    def __init__(self, master):
        self.builder = builder = pygubu.Builder()
        # master.withdraw()
        builder.add_from_file('./mlt.ui')
        self.toplevel = builder.get_object('Toplevel', master)
        # Connect Delete event to a toplevel window
        master.withdraw()
        self.toplevel.protocol("WM_DELETE_WINDOW", self.on_close_window)
        builder.connect_callbacks(self)

        self.model_list = builder.get_object('model_list')
        self.model_list['values'] = ['CNN', 'SVM']
        self.model_list.current(0)
        self.output = builder.get_object('output')
        self.train_size = builder.get_object('tr_size')
        self.test_size = builder.get_object('te_size')
        out = StdoutRedirector(self.output, self.toplevel)
        sys.stdout = out

    def run(self):
        self.toplevel.mainloop()

    def on_close_window(self, event=None):
        print('On close window')
        # Call destroy on toplevel to finish program
        self.toplevel.master.destroy()

    def clear_text(self):
        self.output.delete("1.0", 'end')

    def train(self):
        tr_size = 1000
        te_size = 500
        if self.train_size['text']:
            try:
                tr_size = int(self.train_size['text'])
            except:
                print('Invalid Integer entered for train size "{}", Using default = {}'.format(
                    self.train_size['text'], tr_size))

        if self.test_size['text']:
            try:
                te_size = int(self.test_size['text'])
            except:
                print('Invalid Integer entered for test size "{}", Using default = {}'.format(
                    self.test_size['text'], te_size))

        algo = self.model_list.get()
        print('Starting to train Model using {}!'.format(algo))
        if str(algo).lower() == 'svm':
            svm.svm_train(train_size=tr_size, test_size=te_size)
        else:
            cnn.cnn_train(train_size=tr_size, test_size=te_size)

    def predict_single(self):
        if self.model_list.get() == 'CNN':
            if not cnn.model:
                cnn.load_cnn_model()

            if not cnn.model:
                print('You need to train the model first!')
                return
        else:
            if not svm.model:
                svm.load_svm_model()

            if not svm.model:
                print('You need to train the model first!')
                return

        filename = filedialog.askopenfilename(title='Select Image File to recognize',
                                              filetypes=(("jpeg files", "*.jpg"), ("Bitmaps", "*.bmp")))
        print('Predicting image from file {}'.format(filename))
        if self.model_list.get() == 'CNN':
            cnn.predict_single(filename)
        else:
            svm.predict_single(filename)

    def predict_batch(self):
        if self.model_list.get() == 'CNN':
            if not cnn.model:
                cnn.load_cnn_model()

            if not cnn.model:
                print('You need to train the model first!')
                return
        else:
            if not svm.model:
                svm.load_svm_model()

            if not svm.model:
                print('You need to train the model first!')
                return
        
        filenames = filedialog.askopenfilenames(title='Select Image Files to recognize',
                                                filetypes=(("jpeg files", "*.jpg"), ("Bitmaps", "*.bmp")))
        if self.model_list.get() == 'CNN':
            cnn.predict_multiple(filenames)
        else:
            svm.predict_multiple(filenames)


    def confusion(self):
        test_size = 500
        if self.test_size.get():
            try:
                test_size = int(self.test_size.get())
            except:
                print('Invalid size for Test Size, Using default = 500')

        if self.model_list.get() == 'CNN':
            cnn.show_confusion_matrix(test_size=test_size)
        else:
            svm.show_confusion_matrix(test_size=test_size)


if __name__ == '__main__':
    root = tk.Tk()
    app = Application(root)
    app.run()
