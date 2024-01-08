#!/usr/bin/python3
import pickle
import os
import subprocess
import re
import numpy
import json
import datetime
import time
import sklearn

# Load the classifier and feature list
with open("saved_detector.pkl", "rb") as f:
    clf, features = pickle.load(f)

basepath = "/home/shirsi/lab6/sampless/"

while True:
    try:
        # Enumerate the files in the folder
        for basename in os.listdir(basepath):
            if os.path.isfile(os.path.join(basepath+basename)):
                fullpathname = os.path.join(basepath+basename)

                # Calculate the md5
                process = subprocess.run(["md5sum", fullpathname], check=True, stdout=subprocess.PIPE, universal_newlines=True)
                output = process.stdout
                md5hash = output.split()[0].strip()

                #creates timestamp
                currenttime = datetime.datetime.now()
                timestamp = json.dumps(currenttime.isoformat())
                logMessage = "{ \"timestamp\": "+ timestamp + ", \"md5\": \""+ md5hash +"\""

                try:
                    process = subprocess.run(["./capa", "-v", fullpathname], check=True, capture_output=True, universal_newlines=True)
                    output = process.stdout
                    sections = output.split('\n\n', 1)[1]
                    present_in_file = list()

                    # Scan to see which features are present in the file
                    for section in sections.split('\n\n'):
                        line = section.split('\n')[0]
                        if line != '':
                            present_in_file.append(re.split(r'\(\d', line)[0].strip())

                    #create the vector to predict the file
                    X = list()
                    for feature in features:
                        if feature in present_in_file:
                            X.append(1)
                        else:
                            X.append(0)
                    X =  numpy.reshape(X, (1,-1))

                    # Call the classifier
                    prediction = clf.predict(X)

                    if prediction :
                        logMessage = logMessage + ", \"classification\": \"malware\" }"
                    else:
                        logMessage = logMessage + ", \"classification\": \"benignware\" }"

                except KeyboardInterrupt:
                    logMessage = logMessage + ", \"classification\": \"ND\" } "
                except Exception as e:
                    logMessage = logMessage + ", \"classification\": \"ND\" } "

                # Remove the original file and .viv file
                os.remove(fullpathname)
                if os.path.isfile(fullpathname + ".viv"):
                    os.remove(fullpathname + ".viv")

                with open("detector.log", "a") as f:
                    f.write(json.dumps(logMessage) + "\n")

        time.sleep(5)
    except KeyboardInterrupt:
        exit(0)

