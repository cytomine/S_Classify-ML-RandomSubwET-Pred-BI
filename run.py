# -*- coding: utf-8 -*-

# /**
# * Copyright (c) 2009-2018.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */


__author__          = "Mormont Romain <r.mormont@uliege.be>"
__copyright__       = "Copyright 2010-2018 University of Liège, Belgium, http://www.cytomine.org/"


import os
import numpy as np
from pathlib import Path
from sklearn.externals import joblib
from cytomine.models import *
from cytomine import CytomineJob
from cytomine.utilities.software import setup_classify, parse_domain_list, str2bool


def main(argv):
    with CytomineJob.from_cli(argv) as cj:
        # annotation filtering
        cj.logger.info(str(cj.parameters))

        # use only images from the current project
        cj.parameters.cytomine_id_projects = "{}".format(cj.parameters.cytomine_id_project)

        cj.job.update(progress=1, statuscomment="Preparing execution (creating folders,...).")
        root_path = "/data/" #Path.home()
        image_path, downloaded = setup_classify(
            args=cj.parameters, logger=cj.job_logger(1, 40),
            dest_pattern="{image}_{id}.png", root_path=root_path,
            set_folder="test", showWKT=True
        )

        annotations = [annotation for annotation in downloaded for f in annotation.filenames]
        x = np.array([f for annotation in downloaded for f in annotation.filenames])

        # extract model data from previous job
        cj.job.update(progress=45, statusComment="Extract properties from training job.")
        train_job = Job().fetch(cj.parameters.cytomine_id_job)
        properties = PropertyCollection(train_job).fetch().as_dict()

        binary = str2bool(properties["binary"].value)
        if binary:
            classes = np.array([cj.parameters.cytomine_id_term_negative, cj.parameters.cytomine_id_term_positive])
        else:
            classes = np.array(parse_domain_list(properties["classes"].value))

        # extract model
        cj.job.update(progress=50, statusComment="Download the model file.")
        attached_files = AttachedFileCollection(train_job).fetch()
        model_file = attached_files.find_by_attribute("filename", "model.joblib")
        model_filepath = os.path.join(root_path, "model.joblib")
        model_file.download(model_filepath, override=True)
        pyxit = joblib.load(model_filepath)

        # set n_jobs
        pyxit.base_estimator.n_jobs = cj.parameters.n_jobs
        pyxit.n_jobs = cj.parameters.n_jobs

        cj.job.update(progress=55, statusComment="Predict...")
        if hasattr(pyxit, "predict_proba"):
            probas = pyxit.predict_proba(x)
            y_pred = np.argmax(probas, axis=1)
        else:
            probas = [None] * x.shape[0]
            y_pred = pyxit.predict(x)

        predicted_terms = classes.take(y_pred, axis=0)
        collection = AnnotationCollection()
        for i in cj.monitor(range(x.shape[0]), start=80, end=99, period=0.005, prefix="Uploading predicted terms"):
            annot, term, proba = annotations[i], predicted_terms[i], probas[i]

            parameters = {
                "location": annot.location,
                "id_image": annot.image,
                "id_project": cj.project.id,
                "id_terms": [int(term)]
            }
            if proba is not None:
                parameters["rate"] = float(np.max(proba))
            collection.append(Annotation(**parameters))
        collection.save()
        cj.job.update(status=Job.TERMINATED, status_comment="Finish", progress=100)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
