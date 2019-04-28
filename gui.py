import pygubu
import tkinter as tk
import sys
import svm
import cnn


class StdoutRedirector():
    def __init__(self, text_area, master):
        self.text_area = text_area
        self.master = master

    def write(self,line):
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
                print('Invalid Integer entered for train size "{}", Using default = {}'.format(self.train_size['text'], tr_size))
        
        if self.test_size['text']:
            try:
                te_size = int(self.test_size['text'])
            except:
                print('Invalid Integer entered for test size "{}", Using default = {}'.format(self.test_size['text'], te_size))
                
        algo = self.model_list.get()
        print('Starting to train Model using {}!'.format(algo))
        if str(algo).lower() == 'svm':
            svm.svm_train(train_size=tr_size, test_size=te_size)
        else:
            cnn.cnn_train(train_size=tr_size, test_size=te_size)
    
    def predict_single(self):
        print('Predicting Single!')

    def predict_batch(self):
        print('Predicting Batch!')

    def confusion(self):
        print('Displaying Confusion Matrix!')

if __name__ == '__main__':
    root = tk.Tk()
    app = Application(root)
    app.run()