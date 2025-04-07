git clone https://github.com/mever-team/visloc-estimation.git


# #Descargar pesos aprendidos (meterlo en visloc-estimation)
# !wget https://mever.iti.gr/visloc/back_coll_features.hdf5


#Instalar entorno de Conda

!conda config --add channels conda-forge
!conda config --set channel_priority strict
!conda create -n certhgeoloc python=3.8 pip

!conda activate certhgeoloc

!pip install h5py
!pip install torch torchvision torchaudio
!pip install efficientnet_pytorch==0.7.0

%cd /content/visloc-estimation
!pip install -r requirements.txt





!python inference.py --image_url 'https://thumbs.dreamstime.com/b/tour-eiffel-vu-de-la-rue-%C3%A0-paris-france-trottoir-de-pav%C3%A9-rond-63606834.jpg' --use_cpu