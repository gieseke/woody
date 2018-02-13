#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import os
import gzip
import urllib
import shutil
import h5py
import pandas

def check_and_download(fname, remoteurl="REMOTE_URL"):

    if os.path.isfile(fname) == False:

        if os.path.exists(os.path.join(os.path.dirname(fname), remoteurl)):                
            urlfname = os.path.join(os.path.dirname(fname), remoteurl)
            try:
                with open(urlfname,"r") as f:
                    url = f.readlines()[0].strip()
                    url = os.path.join(url, os.path.basename(fname))
            except Exception as e:
                print("Could not retrieve urlf from file %s" % urlfname)

        elif os.path.exists(fname + ".download"):                
            urlfname = fname + ".download"
            try:
                with open(urlfname,"r") as f:
                    url = f.readlines()[0]
            except Exception as e:
                print("Could not retrieve urlf from file %s" % urlfname)

        else: 
            raise Exception("File and download url do not exist!")

        url = url.strip()
        
        try:
                if url.endswith(".gz"):
                    fname_download = fname + ".gz"
                else:
                    fname_download = fname
                    
                print("Downloading data from %s to %s ..." % (url, fname_download))
                urllib.urlretrieve (url, fname_download)
                
                print("Successfully downloaded the data!")
                if url.endswith(".gz"):
                    print("Extracting zipped file ...")
                    inF = gzip.open(fname_download, 'rb')
                    outF = open(fname, 'wb')
                    outF.write(inF.read())
                    inF.close()
                    outF.close()                    
                    print("Done!")
        except Exception as e:
            print(str(e))
            try:
                # remove incomplete data
                shutil.rmtree(fname)
            except:
                pass
            return False

    return True

def save_to_h5(X, y, fname, compression="lzf"):

    d = os.path.dirname(fname)
    if not os.path.exists(d):
        os.makedirs(d)

    y = y.reshape((len(y), 1))
    
    # create store and data sets
    store = h5py.File(fname, 'w')
    dsetX = store.create_dataset("X", X.shape, compression=compression)
    dsety = store.create_dataset("y", y.shape, compression=compression)
    
    dsetX[:,:] = X
    dsety[:,:] = y
    
    store.close()
    
def save_to_h5pd(X, y, fname, compression="bzip2", complevel=3, delete_before=True):

    d = os.path.dirname(fname)
    if not os.path.exists(d):
        os.makedirs(d)
            
    y = y.reshape((len(y), 1))

    if delete_before == True:
        if os.path.exists(fname):
            os.remove(fname)
        
    df_X = pandas.DataFrame(X, index=range(len(X)))
    df_y = pandas.DataFrame(y, index=range(len(y)))    
    
    df_X.to_hdf(fname, 'X', append=True, complib=compression, complevel=complevel)
    df_y.to_hdf(fname, 'y', append=True, complib=compression, complevel=complevel)
                        
def convert_to_h5pd(reader, fname, transform, compression="bzip2", complevel=3, delete_before=True):

    d = os.path.dirname(fname)
    if not os.path.exists(d):
        os.makedirs(d)

    if delete_before == True:
        if os.path.exists(fname):
            os.remove(fname)
            
    for chunk in reader:
                
        X, y = transform(chunk)            
        y = y.reshape((len(y), 1))
                
        df_X = pandas.DataFrame(X, index=range(len(X)))
        df_y = pandas.DataFrame(y, index=range(len(y)))    
    
        df_X.to_hdf(fname, 'X', append=True, complib=compression, complevel=complevel)
        df_y.to_hdf(fname, 'y', append=True, complib=compression, complevel=complevel)
                        
