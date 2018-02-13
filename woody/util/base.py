#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import os

def makedirs(d):
    """
    """
    
    if not os.path.exists(d):
        os.makedirs(d)
        
def ensure_dir_for_file(f):
    """
    """
    
    d = os.path.dirname(f)
    makedirs(d)  
    
def convert_to_libsvm(ifile_name, ofile_name, counter_print=1000000, label_offset=None):

    orig_labels = []
    new_labels = []
    
    ifile = open(ifile_name, 'r')
    ofile = open(ofile_name, 'w')
    
    # process file line-by-line
    counter = 0
    
    for line in ifile:
    
        new_line = []
    
        if counter % counter_print == 0:
            print("Processing line %i ..." % counter)
            print("orig_labels=" + str(orig_labels))
            print("new_labels=" + str(new_labels))
    
        line = line.split(',')
        
        # append label
        label = line[0]
        orig_labels = list(orig_labels)
        orig_labels.append(label)
        orig_labels = set(orig_labels)
        
        if label_offset is not None:
            label = int(label) + label_offset
        new_labels = list(new_labels)
        new_labels.append(label)
        new_labels = set(new_labels)
            
        new_line.append(str(label))
    
        # append features
        for i, item in enumerate(line[1:]):
            new_item = "%s:%s" % (i+1, item.strip())
            new_line.append(new_item)
    
        new_line = " ".join(new_line)
        new_line += "\n"
    
        ofile.write(new_line)
    
        counter += 1
    
    ifile.close()
    ofile.close()
    
    print("orig_labels=" + str(orig_labels))
    print("new_labels=" + str(new_labels))