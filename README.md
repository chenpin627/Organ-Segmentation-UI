## Run<br>
=>Download segmentation_UI.exe in the URL.<br>
  https://drive.google.com/drive/folders/1bJewOG-J0XeEbx_axwNmowrQGqhiOBBQ?usp=sharing <br>
=>Download the weights of the 6 organs and put them in the weight folder.<br>
=>Run segmentation_UI.exe, the UI interface will appear, as shown in the figure.<br>
=>The steps to use the GUI are listed as follows: <br>
Step 1: Click the “Load Data” button and select the CT image folder of 1 patient pre-downloaded from the PACS server.<br>
Step 2: Click the “Load DicomRT” button and select the patient's DicomRT file.<br>
Step 3: Check the organ to be predicted (can check multiple organs at once).<br>
Step 4: Click “Prediction Start” button to start prediction, and finally a DicomRT file will be generated.<br>
Step 5: Upload the DicomRT file back to the server, then the contours can be read or refined in the treatment planning system.<br>
![GUI](https://user-images.githubusercontent.com/81366172/113831011-c08ad500-97b9-11eb-9c4d-42eea230b92f.jpg)

