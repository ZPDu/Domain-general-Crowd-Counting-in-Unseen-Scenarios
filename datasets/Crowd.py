import os


def _pluck(root):
    img_path = root + '/img'
    ret = [filename for filename in os.listdir(img_path) \
                       if os.path.isfile(os.path.join(img_path, filename))]
    return ret

class CrowdTest(object):
    def __init__(self, root):
        super(CrowdTest, self).__init__()
        self.root = root
        self.train = []
        # self._check_integrity()
        self.load

    @property
    def load(self):

        self.train = _pluck(self.root)
        self.num_train = len(self.train)

        if True:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            print("  train    | {:5d}"
                  .format(self.num_train))
