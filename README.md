## Hardware specifications<br>
=>GPU memory needs more than 5G.
## Run<br>
=>Download segmentation_UI.exe in the URL.<br>
  https://drive.google.com/file/d/1QXz6S7hDmYgBu6imIXlQ7ppC9NabHhTs/view?usp=sharing<br>
=>Download the model of the 6 organs and put them in the Model folder.<br>
=>Run organ_segmentation_GUI.exe, the UI interface will appear, as shown in the figure.<br>
=>The steps to use the GUI are listed as follows: <br>
Step 1: Click the “Load Data” button and select the CT image folder of 1 patient pre-downloaded from the PACS server. The folder contains the patient's CT images and DICOM-RT files.<br>
Step 2: Check the organ to be predicted (can check multiple organs at once or check "Select All" to select all organs).<br>
Step 3: Click “Prediction Start” button to start prediction, and finally a DicomRT file will be generated.<br>
Step 4: Upload the DicomRT file back to the server, then the contours can be read or refined in the treatment planning system.<br>
![GUI](![GUI](https://user-images.githubusercontent.com/81366172/131432015-26a95c4b-3d7e-4831-b437-3f2fae5a700e.PNG)

