try: paraview.simple
except: from paraview.simple import *
paraview.simple._DisableFirstRenderCameraReset()

indata = GetActiveSource()

for i in xrange(20):

    print 'Do threshold'

    Threshold1 = Threshold()
    Threshold1.Scalars = ['POINTS', 'ImageFile']
    Threshold1.ThresholdRange = [3.0391228745293812e-15, 0.1142975240945816]
    Threshold1.ThresholdRange = [0.070000000000000007, 1.0]

    Show()
    Render()

    print 'Delete threshold object'

    Delete(Threshold1)
    SetActiveSource(indata)

    Render()
