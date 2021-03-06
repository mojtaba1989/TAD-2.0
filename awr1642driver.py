import serial
import time
import numpy as np
import platform
import os


class awr1642:
    def __init__(self, configFileName, CLIport_num, Dataport_num):
        self.CLIportNum  = CLIport_num
        self.DataportNum = Dataport_num
        self.CLIport = {}
        self.Dataport = {}
        self.configFileName = configFileName
        self.configParameters = {}
        self.magicOK = 0
        self.dataOK = 0
        self.frameNumber = 0
        self.detObj = {}
        self.config = [line.rstrip('\r\n') for line in open(self.configFileName)]
        self.byteBuffer = np.zeros(2 ** 15, dtype='uint8')
        self.byteBufferLength = 0
        self.magicWord = [2, 1, 4, 3, 6, 5, 8, 7]

    def sendConfig(self):
        for i in self.config:
            if not i == 'sensorStart':
                self.CLIport.write((i + '\n').encode())
                print(i)
            time.sleep(0.01)

    def serrialConfig(self):
        self.CLIport  = {}
        self.Dataport = {}

        if platform.system() == "Windows":
            self.CLIport  = serial.Serial('COM' + self.CLIportNum, 115200, timeout=1)
            self.Dataport = serial.Serial('COM' + self.DataportNum, 921600, timeout=10)

        elif platform.system() == "Darwin":
            self.CLIport = serial.Serial('/dev/tty.usbmodem' + self.CLIportNum, 115200)
            self.Dataport = serial.Serial('/dev/tty.usbmodem' + self.DataportNum, 921600, timeout=1)
        elif platform.system() == "Linux":
            try:
                self.CLIport = serial.Serial('/dev/tty' + self.CLIportNum, 115200)
                self.Dataport = serial.Serial('/dev/tty' + self.DataportNum, 921600, timeout=1)
            except:
                os.system('sudo chmod 666 /dev/tty' + self.CLIportNum)
                os.system('sudo chmod 666 /dev/tty' + self.DataportNum)
                self.CLIport = serial.Serial('/dev/tty' + self.CLIportNum, 115200)
                self.Dataport = serial.Serial('/dev/tty' + self.DataportNum, 921600, timeout=1)
                

    def parseConfigFile(self):
          # Initialize an empty dictionary to store the configuration parameters
        for i in self.config:
            splitWords = i.split(" ")

            # Hard code the number of antennas, change if other configuration is used
            numTxAnt = 2

            # Get the information about the profile configuration
            if "profileCfg" in splitWords[0]:
                startFreq = int(splitWords[2])
                idleTime = int(splitWords[3])
                rampEndTime = float(splitWords[5])
                freqSlopeConst = int(splitWords[8])
                numAdcSamples = int(splitWords[10])
                numAdcSamplesRoundTo2 = 1

                while numAdcSamples > numAdcSamplesRoundTo2:
                    numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2

                digOutSampleRate = int(splitWords[11])

            # Get the information about the frame configuration
            elif "frameCfg" in splitWords[0]:

                chirpStartIdx = int(splitWords[1])
                chirpEndIdx = int(splitWords[2])
                numLoops = int(splitWords[3])
                numFrames = int(splitWords[4])
                framePeriodicity = float(splitWords[5])

        # Combine the read data to obtain the configuration parameters
        numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
        self.configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
        self.configParameters["numRangeBins"] = numAdcSamplesRoundTo2
        self.configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (
                2 * freqSlopeConst * 1e12 * numAdcSamples)
        self.configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (
                2 * freqSlopeConst * 1e12 * self.configParameters["numRangeBins"])
        self.configParameters["dopplerResolutionMps"] = 3e8 / (
                2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * self.configParameters["numDopplerBins"] * numTxAnt)
        self.configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate) / (2 * freqSlopeConst * 1e3)
        self.configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)
        self.configParameters["framePeriodicity"] = framePeriodicity

    def readAndParseData16xx(self):
        # global byteBuffer, byteBufferLength
        # self.byteBuffer = np.zeros(2 ** 15, dtype='uint8')
        # self.byteBufferLength = 0

        # Constants
        OBJ_STRUCT_SIZE_BYTES = 12
        BYTE_VEC_ACC_MAX_SIZE = 2 ** 15
        MMWDEMO_UART_MSG_DETECTED_POINTS = 1
        MMWDEMO_UART_MSG_RANGE_PROFILE = 2
        maxBufferSize = 2 ** 15


        # Initialize variables
        self.magicOK = 0  # Checks if magic number has been read
        self.dataOK = 0  # Checks if the data has been read correctly
        self.frameNumber = 0
        self.detObj = {}


        readBuffer = self.Dataport.read(self.Dataport.in_waiting)
        byteVec = np.frombuffer(readBuffer, dtype='uint8')
        byteCount = len(byteVec)

        # Check that the buffer is not full, and then add the data to the buffer
        if (self.byteBufferLength + byteCount) < maxBufferSize:
            self.byteBuffer[self.byteBufferLength:self.byteBufferLength + byteCount] = byteVec[:byteCount]
            self.byteBufferLength = self.byteBufferLength + byteCount

        # Check that the buffer has some data
        if self.byteBufferLength > 16:

            # Check for all possible locations of the magic word
            possibleLocs = np.where(self.byteBuffer == self.magicWord[0])[0]

            # Confirm that is the beginning of the magic word and store the index in startIdx
            startIdx = []
            for loc in possibleLocs:
                check = self.byteBuffer[loc:loc + 8]
                if np.all(check == self.magicWord):
                    startIdx.append(loc)

            # Check that startIdx is not empty
            if startIdx:

                # Remove the data before the first start index
                if startIdx[0] > 0:
                    self.byteBuffer[:self.byteBufferLength - startIdx[0]] = self.byteBuffer[startIdx[0]:self.byteBufferLength]
                    self.byteBufferLength = self.byteBufferLength - startIdx[0]

                # Check that there have no errors with the byte buffer length
                if self.byteBufferLength < 0:
                    self.byteBufferLength = 0

                # word array to convert 4 bytes to a 32 bit number
                word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

                # Read the total packet length
                totalPacketLen = np.matmul(self.byteBuffer[12:12 + 4], word)

                # Check that all the packet has been read
                if (self.byteBufferLength >= totalPacketLen) and (self.byteBufferLength != 0):
                    self.magicOK = 1

        # If magicOK is equal to 1 then process the message
        if self.magicOK:
            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

            # Initialize the pointer index
            idX = 0

            # Read the header
            magicNumber = self.byteBuffer[idX:idX + 8]
            idX += 8
            version = format(np.matmul(self.byteBuffer[idX:idX + 4], word), 'x')
            idX += 4
            totalPacketLen = np.matmul(self.byteBuffer[idX:idX + 4], word)
            idX += 4
            platform = format(np.matmul(self.byteBuffer[idX:idX + 4], word), 'x')
            idX += 4
            self.frameNumber = np.matmul(self.byteBuffer[idX:idX + 4], word)
            idX += 4
            timeCpuCycles = np.matmul(self.byteBuffer[idX:idX + 4], word)
            idX += 4
            numDetectedObj = np.matmul(self.byteBuffer[idX:idX + 4], word)
            idX += 4
            numTLVs = np.matmul(self.byteBuffer[idX:idX + 4], word)
            idX += 4
            subFrameNumber = np.matmul(self.byteBuffer[idX:idX + 4], word)
            idX += 4

            # Read the TLV messages
            for tlvIdx in range(numTLVs):

                # word array to convert 4 bytes to a 32 bit number
                word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

                # Check the header of the TLV message
                tlv_type = np.matmul(self.byteBuffer[idX:idX + 4], word)
                idX += 4
                tlv_length = np.matmul(self.byteBuffer[idX:idX + 4], word)
                idX += 4

                # Read the data depending on the TLV message
                if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:

                    # word array to convert 4 bytes to a 16 bit number
                    word = [1, 2 ** 8]
                    tlv_numObj = np.matmul(self.byteBuffer[idX:idX + 2], word)
                    idX += 2
                    tlv_xyzQFormat = 2 ** np.matmul(self.byteBuffer[idX:idX + 2], word)
                    idX += 2

                    # Initialize the arrays
                    rangeIdx = np.zeros(tlv_numObj, dtype='int16')
                    dopplerIdx = np.zeros(tlv_numObj, dtype='int16')
                    peakVal = np.zeros(tlv_numObj, dtype='int16')
                    x = np.zeros(tlv_numObj, dtype='int16')
                    y = np.zeros(tlv_numObj, dtype='int16')
                    z = np.zeros(tlv_numObj, dtype='int16')

                    for objectNum in range(tlv_numObj):
                        # Read the data for each object
                        rangeIdx[objectNum] = np.matmul(self.byteBuffer[idX:idX + 2], word)
                        idX += 2
                        dopplerIdx[objectNum] = np.matmul(self.byteBuffer[idX:idX + 2], word)
                        idX += 2
                        peakVal[objectNum] = np.matmul(self.byteBuffer[idX:idX + 2], word)
                        idX += 2
                        x[objectNum] = np.matmul(self.byteBuffer[idX:idX + 2], word)
                        idX += 2
                        y[objectNum] = np.matmul(self.byteBuffer[idX:idX + 2], word)
                        idX += 2
                        z[objectNum] = np.matmul(self.byteBuffer[idX:idX + 2], word)
                        idX += 2

                    # Make the necessary corrections and calculate the rest of the data
                    rangeVal = rangeIdx * self.configParameters["rangeIdxToMeters"]
                    dopplerIdx[dopplerIdx > (self.configParameters["numDopplerBins"] / 2 - 1)] = dopplerIdx[dopplerIdx > (
                                self.configParameters["numDopplerBins"] / 2 - 1)] - 65535
                    dopplerVal = dopplerIdx * self.configParameters["dopplerResolutionMps"]
                    # x[x > 32767] = x[x > 32767] - 65536
                    # y[y > 32767] = y[y > 32767] - 65536
                    # z[z > 32767] = z[z > 32767] - 65536
                    x = x / tlv_xyzQFormat
                    y = y / tlv_xyzQFormat
                    z = z / tlv_xyzQFormat

                    # Store the data in the detObj dictionary
                    self.detObj = {"numObj": tlv_numObj, "rangeIdx": rangeIdx, "range": rangeVal, "dopplerIdx": dopplerIdx, \
                              "doppler": dopplerVal, "peakVal": peakVal, "x": x, "y": y, "z": z}

                    self.dataOK = 1

                    # print(detObj['range'].mean())

                elif tlv_type == MMWDEMO_UART_MSG_RANGE_PROFILE:
                    idX += tlv_length

            # Remove already processed data
            if idX > 0 and self.dataOK == 1:
                shiftSize = idX

                self.byteBuffer[:self.byteBufferLength - shiftSize] = self.byteBuffer[shiftSize:self.byteBufferLength]
                self.byteBufferLength = self.byteBufferLength - shiftSize

                # Check that there are no errors with the buffer length
                if self.byteBufferLength < 0:
                    self.byteBufferLength = 0

    def update(self):
        self.readAndParseData16xx()

    def close(self):
        self.CLIport.write(('sensorStop\n').encode())
        time.sleep(0.01)
        print('sensorStop\n')
        self.CLIport.close()
        time.sleep(0.01)
        self.Dataport.close()
        time.sleep(0.01)
        self.CLIport = {}
        self.Dataport = {}

    def sensorSetup(self):
        self.serrialConfig()
        self.sendConfig()
        self.parseConfigFile()

    def setDetectionThreashold(self, new_threshold):
        idx = [i for i, s in enumerate(self.config) if "cfarCfg -1 0" in s][0]
        temp = self.config[idx].split(" ")
        temp[8] = str(int(new_threshold))
        self.config[idx] = ' '.join(temp)
        time.sleep(.005)
        self.CLIport.write((self.config[idx] + '\n').encode())
        print(self.config[idx])

    def setNoiseAveragingWindow(self, numSamples):
        numSamplesRound2 = 1
        while numSamplesRound2 < numSamples:
            numSamplesRound2 = numSamplesRound2 * 2

        idx = [i for i, s in enumerate(self.config) if "cfarCfg -1 0" in s][0]
        temp = self.config[idx].split(" ")
        temp[4] = str(numSamplesRound2)
        temp[5] = str(int(numSamplesRound2/2))
        temp[6] = str(np.log2(numSamplesRound2).astype(int) + 1)
        self.config[idx] = ' '.join(temp)
        time.sleep(.01)
        self.CLIport.write((self.config[idx] + '\n').encode())
        print(self.config[idx])

    def setRangePeakGrouping(self, state='enable'):
        idx = [i for i, s in enumerate(self.config) if "peakGrouping" in s][0]
        temp = self.config[idx].split(" ")
        if state in ['enable', 'e']:
            temp[3] = str(int(1))
        else:
            temp[3] = str(int(0))
        self.config[idx] = ' '.join(temp)
        self.CLIport.write((self.config[idx] + '\n').encode())
        time.sleep(0.01)
        print(self.config[idx])

    def setMaxRange(self, range):
        self.CLIport.write(('sensorStop\n').encode())
        time.sleep(0.01)
        print('sensorStop\n')
        idx = [i for i, s in enumerate(self.config) if "frameCfg" in s][0]
        temp = self.config[idx].split(" ")
        frameTime = np.min([float(temp[5]) * 1e3, 50e3])
        idx = [i for i, s in enumerate(self.config) if "profileCfg" in s][0]
        temp = self.config[idx].split(" ")
        freqSlopeConst   = int(np.max([20, np.ceil(240/range/5) * 5]))
        rampEndTime      = np.round(4000 / freqSlopeConst, decimals=2)
        adcStartTime     = int(temp[4])
        txStartTime      = int(temp[9])
        idleTime         = int(np.round(frameTime/64-rampEndTime-adcStartTime-txStartTime, decimals=0))
        adcSamplingTime  = rampEndTime - adcStartTime - txStartTime
        digOutSampleRate = int(np.round(range * 2 * freqSlopeConst / .24, decimals=0))
        numAdcSamples    = int(np.floor(digOutSampleRate * adcSamplingTime / 4e3) * 4)
        temp[3]  = str(idleTime)
        temp[5]  = str(rampEndTime)
        temp[8]  = str(freqSlopeConst)
        temp[10] = str(numAdcSamples)
        temp[11] = str(digOutSampleRate)
        self.config[idx] = ' '.join(temp)
        self.CLIport.write((self.config[idx] + '\n').encode())
        print(self.config[idx])
        self.parseConfigFile()

    def setSampleRate(self, rate):
        self.CLIport.write(('sensorStop\n').encode())
        time.sleep(0.01)
        print('sensorStop\n')
        idx = [i for i, s in enumerate(self.config) if "frameCfg" in s][0]
        temp = self.config[idx].split(" ")
        temp[5] = str(np.round(1e3 / rate, decimals=2))
        self.configParameters["framePeriodicity"] = float(temp[5])
        self.config[idx] = ' '.join(temp)
        self.CLIport.write((self.config[idx] + '\n').encode())
        print(self.config[idx])

    def optimize(self, range_interval, eval_range):
        detection_range = np.linspace(range_interval[0], range_interval[1], 10)
        score = np.zeros(10)

        for i in range(10):
            self.byteBuffer = np.zeros(2 ** 15, dtype='uint8')
            self.byteBufferLength = 0
            self.setMaxRange(detection_range[i])
            time.sleep(.01)
            self.CLIport.write(('sensorStart\n').encode())
            time.sleep(.01)
            n = 90
            radar = np.zeros((n, n, 20))
            for itr in range(20):
                loopStartTime = time.time()
                self.update()
                while not self.dataOK and time.time() - loopStartTime > self.configParameters[
                    "framePeriodicity"] / 1000:
                    self.update()
                    time.sleep(.01)
                if not self.dataOK:
                    self.Dataport.reset_output_buffer()
                    self.Dataport.reset_input_buffer()
                    self.byteBuffer = np.zeros(2 ** 15, dtype='uint8')
                    self.byteBufferLength = 0
                    time.sleep(.01)
                else:
                    print(self.dataOK, self.detObj)
                    radar = self.heat_map(radar, self.detObj["x"], self.detObj["y"], self.detObj["peakVal"],
                                                [-eval_range, eval_range], [0, eval_range], xbinnum=n, ybinnum=n)
                time.sleep(np.max([0,
                                   self.configParameters["framePeriodicity"] / 1000 - (time.time() - loopStartTime)]))
            score[i] = np.mean(np.mean(np.mean(radar)))
            self.Dataport.reset_output_buffer()
            self.Dataport.reset_input_buffer()
            self.CLIport.write(('sensorStop\n').encode())
            time.sleep(.01)
        print('range', detection_range, 'score', score)

    def run(self, end=np.nan):
        self.Dataport.reset_output_buffer()
        self.Dataport.reset_input_buffer()
        self.byteBuffer = np.zeros(2 ** 15, dtype='uint8')
        self.byteBufferLength = 0
        self.CLIport.write(('sensorStart\n').encode())
        time.sleep(.01)
        try:
            itt = 0
            while True:
                if not np.isnan(end):
                    if itt <= end:
                        itt += 1
                    else:
                        break
                loopStartTime = time.time()
                self.update()
                while not self.dataOK and time.time()-loopStartTime > self.configParameters["framePeriodicity"]/1000:
                    self.update()
                    time.sleep(.01)
                if not self.dataOK:
                    self.Dataport.reset_output_buffer()
                    self.Dataport.reset_input_buffer()
                    self.byteBuffer = np.zeros(2 ** 15, dtype='uint8')
                    self.byteBufferLength = 0
                    time.sleep(.01)
                print(self.dataOK, self.detObj)
                time.sleep(self.configParameters["framePeriodicity"]/1000 - (time.time() - loopStartTime))
        except KeyboardInterrupt:
            self.Dataport.reset_output_buffer()
            self.Dataport.reset_input_buffer()
            self.CLIport.write(('sensorStop\n').encode())

    def runPlot(self):
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtGui
        max_range = self.configParameters["maxRange"]
        pg.setConfigOption('background', 'w')
        win = pg.GraphicsWindow(title="2D scatter plot")
        p = win.addPlot()
        p.setXRange(-max_range, max_range)
        p.setYRange(0, max_range)
        p.setLabel('left', text='Y position (m)')
        p.setLabel('bottom', text='X position (m)')
        s = p.plot([], [], pen=None, symbol='o')

        self.Dataport.reset_output_buffer()
        self.byteBuffer = np.zeros(2 ** 15, dtype='uint8')
        self.byteBufferLength = 0
        self.CLIport.write(('sensorStart\n').encode())
        time.sleep(.01)

        try:
            while True:
                loopStartTime = time.time()
                self.update()
                i = 0
                while not self.dataOK and i < 10:
                    self.update()
                    time.sleep(.005)
                    i += 1
                if self.dataOK:
                    s.setData(self.detObj["x"], self.detObj["y"])
                else:
                    self.Dataport.reset_output_buffer()
                    self.Dataport.reset_input_buffer()
                    self.byteBuffer = np.zeros(2 ** 15, dtype='uint8')
                    self.byteBufferLength = 0
                QtGui.QApplication.processEvents()
                time.sleep(np.max([0,
                                   self.configParameters["framePeriodicity"] / 1000 - (time.time() - loopStartTime)]))
        except KeyboardInterrupt:
            self.Dataport.reset_output_buffer()
            self.Dataport.reset_input_buffer()
            self.CLIport.write(('sensorStop\n').encode())
            time.sleep(.01)
            win.close()
        win.close()

    def heat_map(self, tabold, xr, yr, zr, xlim, ylim, xc=np.nan, yc=np.nan, xbinnum=100, ybinnum=100):

        x_edges = np.linspace(xlim[0], xlim[1], xbinnum)
        y_edges = np.linspace(ylim[0], ylim[1], ybinnum)

        try:
            valid_list = np.logical_and(
                np.logical_and(xr >= xlim[0], xr <= xlim[1]),
                np.logical_and(yr >= ylim[0], yr <= ylim[1]))

            xr = xr[valid_list]
            yr = yr[valid_list]
            zr = zr[valid_list]

            indx = np.digitize(xr, x_edges)
            indy = np.digitize(yr, y_edges)

            xr = x_edges[indx - 1]
            yr = y_edges[indy - 1]

            indx = np.digitize(xc, x_edges)
            indy = np.digitize(yc, y_edges)

            xc = x_edges[indx - 1]
            yc = y_edges[indy - 1]

            tab = np.zeros([xbinnum, ybinnum])

            for i in range(len(xr)):
                tab[np.where(x_edges == xr[i]), np.where(y_edges == yr[i])] = + zr[i]

            try:
                for i in range(len(xc)):
                    tab[np.where(x_edges == xc[i]), np.where(y_edges == yc[i])] = + 1
            except:
                pass

            tabold = np.append(tab.reshape(xbinnum, ybinnum, 1), tabold, axis=-1)
            tabold = np.delete(tabold, -1, axis=-1)

            return tabold
        except:
            pass


if __name__ == '__main__':
    driver = awr1642("profile.cfg", "ACM0", "ACM1")
    driver.sensorSetup()
    time.sleep(.01)
    # driver.optimize((5, 15) ,10)
    # driver.setMaxRange(15)
    # time.sleep(.01)
    driver.run(20)
    # driver.setNoiseAveragingWindow(32)
    # time.sleep(1)
    # driver.setRangePeakGrouping('dis')
    # driver.run()
    # # time.sleep(1)
    # driver.setRangePeakGrouping('enable')
    # time.sleep(5)
    # driver.run()
    driver.close()
