import sys
sys.path.append(r"C:\Program Files (x86)\CST STUDIO SUITE 2023\AMD64\python_cst_libraries")
import cst
import cst.results as cstr
import cst.interface as csti
import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as scimage
from scipy.fft import fft, fftfreq
from PIL import Image, ImageDraw, ImageFont
import matplotlib.colors as colors
from math import ceil, sqrt


class CSTInterface:
    def __init__(self, fname):
        self.full_path = os.getcwd() + f"\{fname}"
        self.opencst()

    def opencst(self):
        print("CST opening...")
        allpids = csti.running_design_environments()
        open = False
        for pid in allpids:
            self.de = csti.DesignEnvironment.connect(pid)
            # self.de.set_quiet_mode(True) # suppress message box
            print(f"Opening {self.full_path}...")
            self.prj = self.de.open_project(self.full_path)
            open = True
            print(f"{self.full_path} open")
            break
        if not open:
            print("File path not found in current design environment...")
            print("Opening new design environment...")
            self.de = csti.DesignEnvironment.new()
            # self.de.set_quiet_mode(True) # suppress message box
            self.prj = self.de.open_project(self.full_path)
            open = True
            print(f"{self.full_path} open")
        # I wish to create project if not created, but I don't know how to.
        # Therefore, the user need to create *.cst file in advance

    def read(self, result_item):
        results = cstr.ProjectFile(self.full_path, True) #bool: allow interactive
        try:
            res = results.get_3d().get_result_item(result_item)
            res = res.get_data()
        except:
            print("No result item. Available result items listed below")
            print(results.get_3d().get_tree_items())
            res = None
        return res

    def save(self):
        self.prj.modeler.full_history_rebuild() 
        #update history, might discard changes if not added to history list
        self.prj.save()

    def close(self):
        self.de.close()

    def excute_vba(self,  command):
        command = "\n".join(command)
        vba = self.prj.schematic
        res = vba.execute_vba_code(command)
        return res

    def create_para(self,  para_name, para_value): #create or change are the same
        command = ['Sub Main', 'StoreDoubleParameter("%s", "%.4f")' % (para_name, para_value),
                'RebuildOnParametricChange(False, True)', 'End Sub']
        res = self.excute_vba (command)
        return command
    
    def create_shape(self, index, xmin, xmax, ymin, ymax, hc): #create or change are the same
        command = ['With Brick', '.Reset ', f'.Name "solid{index}" ', 
                   '.Component "component2" ', f'.Material "material{index}" ', 
                   f'.Xrange "{xmin}", "{xmax}" ', f'.Yrange "{ymin}", "{ymax}" ', 
                   f'.Zrange "0", "{hc}" ', '.Create', 'End With']
        return command
        # command = "\n".join(command)
        # self.prj.modeler.add_to_history(f"solid{index}",command)
    
    def create_cond_material(self, index, sigma, type="Lossy metal"): #create or change are the same
        command = ['With Material', '.Reset ', f'.Name "material{index}"', 
                #    '.Folder ""', '.Rho "8930"', '.ThermalType "Normal"', 
                #    '.ThermalConductivity "401"', '.SpecificHeat "390", "J/K/kg"', 
                #    '.DynamicViscosity "0"', '.UseEmissivity "True"', '.Emissivity "0"', 
                #    '.MetabolicRate "0.0"', '.VoxelConvection "0.0"', 
                #    '.BloodFlow "0"', '.MechanicsType "Isotropic"', 
                #    '.YoungsModulus "120"', '.PoissonsRatio "0.33"', 
                #    '.ThermalExpansionRate "17"', '.IntrinsicCarrierDensity "0"', 
                   '.FrqType "all"', f'.Type "{type}"', 
                   '.MaterialUnit "Frequency", "GHz"', '.MaterialUnit "Geometry", "mm"', 
                   '.MaterialUnit "Time", "ns"', '.MaterialUnit "Temperature", "Celsius"', 
                   '.Mu "1"', f'.Sigma "{sigma}"', 
                   '.LossyMetalSIRoughness "0.0"', '.ReferenceCoordSystem "Global"', 
                   '.CoordSystemType "Cartesian"', '.NLAnisotropy "False"', 
                   '.NLAStackingFactor "1"', '.NLADirectionX "1"', '.NLADirectionY "0"', 
                   '.NLADirectionZ "0"', '.Colour "0", "1", "1" ', '.Wireframe "False" ', 
                   '.Reflection "False" ', '.Allowoutline "True" ', 
                   '.Transparentoutline "False" ', '.Transparency "0" ', 
                   '.Create', 'End With']
        return command
        # command = "\n".join(command)
        # self.prj.modeler.add_to_history(f"material{index}",command)

    def start_simulate(self, plane_wave_excitation=False):
        try: # problems occur with extreme conditions
            if plane_wave_excitation:
                command = ['Sub Main', 'With Solver', 
                '.StimulationPort "Plane wave"', 'End With', 'End Sub']
                self.excute_vba(command)
                print("Plane wave excitation = True")
            # one actually should not do try-except otherwise severe bug may NOT be detected
            model = self.prj.modeler
            model.run_solver()
        except Exception as e: pass
    
    def set_plane_wave(self):  # doesn't update history, disappear after save but remain after simulation
        command = ['Sub Main', 'With PlaneWave', '.Reset ', 
                   '.Normal "0", "0", "-1" ', '.EVector "1", "0", "0" ', 
                   '.Polarization "Linear" ', '.ReferenceFrequency "2" ', 
                   '.PhaseDifference "-90.0" ', '.CircularDirection "Left" ', 
                   '.AxialRatio "0.0" ', '.SetUserDecouplingPlane "False" ', 
                   '.Store', 'End With', 'End Sub']
        res = self.excute_vba(command)
        return res
    
    def set_excitation(self, filePath): # doesn't update history, disappear after save but remain after simulation. 
        # set .UseCopyOnly to false otherwise CST read cache
        command = ['Sub Main', 'With TimeSignal ', '.Reset ', 
                   '.Name "signal1" ', '.SignalType "Import" ', 
                   '.ProblemType "High Frequency" ', 
                   f'.FileName "{filePath}" ', 
                   '.Id "1"', '.UseCopyOnly "false" ', '.Periodic "False" ', 
                   '.Create ', '.ExcitationSignalAsReference "signal1", "High Frequency"',
                   'End With', 'End Sub']
        res = self.excute_vba(command)
        return res
    
    def delete_plane_wave(self):
        command = ['Sub Main', 'PlaneWave.Delete', 'End Sub']
        res = self.excute_vba(command)
        return res
    
    def delete_signal1(self):
        command = ['Sub Main', 'With TimeSignal', 
     '.Delete "signal1", "High Frequency" ', 'End With', 'End Sub']
        res = self.excute_vba(command)
        return res
    
    def set_port(self, point1, point2): # Not a robust piece of code, but anyway
        command = ['Sub Main', 'Pick.PickEdgeFromId "component1:feed", "1", "1"', 
                   'Pick.PickEdgeFromId "component1:coaxouter", "1", "1"', 
                   'With DiscreteFacePort ', '.Reset ', '.PortNumber "1" ', 
                   '.Type "SParameter"', '.Label ""', '.Folder ""', '.Impedance "50.0"', 
                   '.VoltageAmplitude "1.0"', '.CurrentAmplitude "1.0"', '.Monitor "True"', 
                   '.CenterEdge "True"', f'.SetP1 "True", "{point1[0]}", "{point1[1]}", "{point1[2]}"', 
                   f'.SetP2 "True", "{point2[0]}", "{point2[1]}", "{point2[2]}"', '.LocalCoordinates "False"', 
                   '.InvertDirection "False"', '.UseProjection "False"', 
                   '.ReverseProjection "False"', '.FaceType "Linear"', '.Create ', 
                   'End With', 'End Sub']
        res = self.excute_vba(command)
        return res
    
    def delete_port(self):
        command = ['Sub Main', 'Port.Delete "1"', 'End Sub']
        res = self.excute_vba(command)
        return res
    
    def export_E_field(self, outputPath, resultPath, time_end, time_step, d_step):
        total_samples = int(time_end/time_step)
        command = ['Sub Main',
        'SelectTreeItem  ("%s")' % resultPath, 
        'With ASCIIExport', '.Reset',
        f'.FileName ("{outputPath}")',
        f'.SetSampleRange(0, {total_samples})',
        '.Mode ("FixedWidth")', f'.Step ({d_step})',
        '.Execute', 'End With', 'End Sub']
        res = self.excute_vba(command)
        return res
    
    def export_power(self, outputPath, resultPath, time_end, time_step):
        total_samples = int(time_end/time_step)
        command = ['Sub Main',
        f'SelectTreeItem  ("{resultPath}")', 
        'With ASCIIExport', '.Reset',
        f'.FileName ("{outputPath}")',
        f'.SetSampleRange(0, {total_samples})',
        '.StepX (4)', '.StepY (4)',
        '.Execute', 'End With', 'End Sub']
        res = self.excute_vba(command)
        return res
    
    def delete_results(self):
        command = ['Sub Main',
        'DeleteResults', 'End Sub']
        res = self.excute_vba(command)
        return res
 

class Controller(CSTInterface):
    def __init__(self, fname):
        super().__init__(fname)

    def set_PIFA(self):
        print("Setting PIFA...")
        # Create ground, substrate, feed, and port
        ground = ['Component.New "component1"', 'Component.New "component2"',
                   'With Brick', '.Reset ', 
                   '.Name "ground" ', '.Component "component1" ', 
                   '.Material "Copper (annealed)" ', f'.Xrange "{-self.Lg/2}", "{self.Lg/2}" ', 
                   f'.Yrange "{-self.Wg/2}", "{self.Wg/2}" ', f'.Zrange "{-self.hc-self.hs}", "{-self.hs}" ', '.Create', 'End With']
        substrate = ['With Material', '.Reset', '.Name "FR-4 (loss free)"', 
                   '.Folder ""', '.FrqType "all"', '.Type "Normal"', 
                   '.SetMaterialUnit "GHz", "mm"', '.Epsilon "4.3"', '.Mu "1.0"', 
                   '.Kappa "0.0"', '.KappaM "0.0"', 
                   '.TanDM "0.0"', '.TanDMFreq "0.0"', '.TanDMGiven "False"', 
                   '.TanDMModel "ConstKappa"', '.DispModelEps "None"', 
                   '.DispModelMu "None"', '.DispersiveFittingSchemeEps "General 1st"', 
                   '.DispersiveFittingSchemeMu "General 1st"', 
                   '.UseGeneralDispersionEps "False"', '.UseGeneralDispersionMu "False"', 
                   '.Rho "0.0"', '.ThermalType "Normal"', '.ThermalConductivity "0.3"', 
                   '.SetActiveMaterial "all"', '.Colour "0.94", "0.82", "0.76"', 
                   '.Wireframe "False"', '.Transparency "0"', '.Create', 'End With',
                   'With Brick', '.Reset ', '.Name "substrate" ', 
                   '.Component "component1" ', '.Material "FR-4 (loss free)" ', 
                   f'.Xrange "{-self.Lg/2}", "{self.Lg/2}" ', f'.Yrange "{-self.Wg/2}", "{self.Wg/2}" ', 
                   f'.Zrange "{-self.hs}", "0" ', '.Create', 'End With ']
        ground_sub = ['With Cylinder ', '.Reset ', '.Name "sub" ', '.Component "component1" ', 
                   '.Material "Copper (annealed)" ', f'.OuterRadius "{self.hs}" ', 
                   '.InnerRadius "0.0" ', '.Axis "z" ', f'.Zrange "{-self.hc-self.hs}", "{-self.hs}" ', 
                   f'.Xcenter "{self.feedx}" ', f'.Ycenter "{self.feedy}" ', '.Segments "0" ', '.Create ', 
                   'End With', 'Solid.Subtract "component1:ground", "component1:sub"']
        substrate_sub = ['With Cylinder ', '.Reset ', '.Name "feedsub" ', 
                   '.Component "component1" ', '.Material "FR-4 (loss free)" ', 
                   f'.OuterRadius "{self.hs/2-0.1}" ', '.InnerRadius "0.0" ', '.Axis "z" ', 
                   f'.Zrange "{-self.hs}", "0" ', f'.Xcenter "{self.feedx}" ', f'.Ycenter "{self.feedy}" ', 
                   '.Segments "0" ', '.Create ', 'End With', 
                   'Solid.Subtract "component1:substrate", "component1:feedsub"'] 
        feed = ['With Cylinder ', '.Reset ', '.Name "feed" ', '.Component "component1" ', 
                   '.Material "PEC" ', f'.OuterRadius "{self.hs/2-0.1}" ', '.InnerRadius "0.0" ', 
                   '.Axis "z" ', f'.Zrange "{-5-self.hc-self.hs}", "{self.hc}" ', f'.Xcenter "{self.feedx}" ', 
                   f'.Ycenter "{self.feedy}" ', '.Segments "0" ', '.Create ', 'End With']
        coax = ['With Cylinder ', '.Reset ', '.Name "coax" ', '.Component "component1" ', 
                   '.Material "Vacuum" ', f'.OuterRadius "{self.hs-0.01}" ', f'.InnerRadius "{self.hs/2-0.1}" ', 
                   '.Axis "z" ', f'.Zrange "{-5-self.hc-self.hs}", "{-self.hc-self.hs}" ', f'.Xcenter "{self.feedx}" ', 
                   f'.Ycenter "{self.feedy}" ', '.Segments "0" ', '.Create ', 'End With', 
                   'With Cylinder ', '.Reset ', '.Name "coaxouter" ', 
                   '.Component "component1" ', '.Material "PEC" ', f'.OuterRadius "{self.hs}" ', 
                   f'.InnerRadius "{self.hs-0.01}" ', '.Axis "z" ', f'.Zrange "{-5-self.hc-self.hs}", "{-self.hc-self.hs}" ', 
                   f'.Xcenter "{self.feedx}" ', f'.Ycenter "{self.feedy}" ', '.Segments "0" ', '.Create ', 
                   'End With']
        command = ground + substrate + ground_sub + substrate_sub + feed + coax
        command = "\n".join(command)
        self.prj.modeler.add_to_history("initialize",command)
        self.save()
        print("PIFA set")

    def set_surroundng_blocks(self):
        print("Setting surrounding_blocks...")
        # Initialize surrounding_blocks with uniform conductivity
        cond = np.zeros(8)
        # Define materials first
        self.update_distribution(cond)
        command = []
        # Define shape and index based on materials
        postion = [position1(x,y), postion2(x,y), ...pos8]
        for index, pos in enumerate(postion): 
            midpoint = (self.Ld/2, self.Wd/2)
            xi = index%nx
            yi = index//nx
            xmin = xi*self.d-midpoint[0]
            xmax = xmin+self.d
            ymin = yi*self.d-midpoint[1]
            ymax = ymin+self.d
            command += self.create_shape(index, xmin, xmax, ymin, ymax, self.hc)
        command = "\n".join(command)
        self.prj.modeler.add_to_history("surrounding_blocks",command)
        self.save()
        print("Surrounding_blocks set")

    def update_distribution(self, cond):
        print("Material distribution updating...")
        command_material = []
        for index, sigma in enumerate(cond):
            if sigma == 1: command_material += self.create_cond_material(index, sigma, "PEC")
            elif sigma == 0: command_material += self.create_cond_material(index, sigma, "Normal")
            else: print("undefined material")
        command_material = "\n".join(command_material)
        self.prj.modeler.add_to_history("material update",command_material)
        print("Material distribution updated")