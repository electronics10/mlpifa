import sys
sys.path.append(r"C:\Program Files (x86)\CST STUDIO SUITE 2023\AMD64\python_cst_libraries")
import cst
import cst.results as cstr
import cst.interface as csti
import os
import numpy as np
import pandas as pd
from settings import*

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
            try: self.prj = self.de.open_project(self.full_path)
            except: 
                print(f"Creating new project {self.full_path}")
                self.prj = self.de.new_mws()
                self.prj.save(self.full_path)
            open = True
            print(f"{self.full_path} open")
            break
        if not open:
            print("File path not found in current design environment...")
            print("Opening new design environment...")
            self.de = csti.DesignEnvironment.new()
            # self.de.set_quiet_mode(True) # suppress message box
            try: self.prj = self.de.open_project(self.full_path)
            except: 
                print(f"Creating new project {self.full_path}")
                self.prj = self.de.new_mws()
                self.prj.save(self.full_path)
            open = True
            print(f"{self.full_path} open")

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

    def start_simulate(self):
        try: # unknown problems occur under extreme conditions
            print("Solving...")
            model = self.prj.modeler
            model.run_solver()
            print("Solved")
        except Exception as e: pass


class MLPIFA(CSTInterface):
    def __init__(self, fname):
        super().__init__(fname)
        self.blocks_num = BLOCKS_NUM
        self.block_len = BLOCK_LEN
        self.block_sep = BLOCK_SEPERATION
        self.fx_min = FEEDX_MIN 
        self.fx_max = FEEDX_MAX
        self.desRegionX = REGIONX
        self.desRegionY = REGIONY
        self.parameters = {"pin_dis":INIT_P1, "patch_len":INIT_P2, "pin_width":INIT_P3} # 2.45 GHz PIFA

    def excute_vba(self,  command):
        command = "\n".join(command)
        vba = self.prj.schematic
        vba.execute_vba_code(command)

    def update_distribution(self, binary_sequence):
        print("Material distribution updating...")
        command_material = []
        for index, sigma in enumerate(binary_sequence):
            if sigma == 1: command_material += self.create_PEC_material(index)
            elif sigma == 0: command_material += self.create_air_material(index)
            else: print("undefined material")
        command_material = "\n".join(command_material)
        self.prj.modeler.add_to_history("material update",command_material)
        print("Material distribution updated")

    def create_PEC_material(self, index): #create or change are the same
        command = ['With Material', '.Reset ', f'.Name "material{index}"', 
                   '.FrqType "all"', '.Type "PEC"', 
                   '.MaterialUnit "Frequency", "GHz"', '.MaterialUnit "Geometry", "mm"', 
                   '.MaterialUnit "Time", "ns"', '.MaterialUnit "Temperature", "Celsius"',  
                   '.Create', 'End With']
        return command
    
    def create_air_material(self, index): #create or change are the same
        command = ['With Material', '.Reset ', f'.Name "material{index}"',  
                   '.FrqType "all"', '.Type "Normal"', 
                   '.MaterialUnit "Frequency", "GHz"', '.MaterialUnit "Geometry", "mm"', 
                   '.MaterialUnit "Time", "ns"', '.MaterialUnit "Temperature", "Celsius"', 
                   '.Epsilon "1"', '.Mu "1"', '.Create', 'End With']
        return command
    
    def create_blocks(self, index, xmin, xmax, ymin, ymax): #create or change are the same
        command = ['With Brick', '.Reset ', f'.Name "solid{index}" ', 
                   '.Component "component2" ', f'.Material "material{index}" ', 
                   f'.Xrange "{xmin}", "{xmax}" ', f'.Yrange "{ymin}", "{ymax}" ', 
                   f'.Zrange "0", "0.035" ', '.Create', 'End With']
        return command

    def create_parameters(self,  para_name, para_value): #create or change are the same
        command = ['Sub Main', 'StoreDoubleParameter("%s", "%.4f")' % (para_name, para_value),
                'RebuildOnParametricChange(False, True)', 'End Sub']
        self.excute_vba(command)
    
    def delete_results(self):
        command = ['Sub Main', 'DeleteResults', 'End Sub']
        self.excute_vba(command)

    def set_environment(self):
        # Initiate parameters
        for name, val in self.parameters.items(): 
            self.create_parameters(name, val)
        self.create_parameters('fx', (self.fx_min+self.fx_max)/2) # create initial fx
        print("Parameters created.")
        # Create PIFA and surroundings
        command = ['Sub Main']
        self.set_PIFA()
        self.set_blocks()
        command.append('End Sub')
        self.excute_vba(command)

    def set_PIFA(self):
        y_offset = OFFSETY
        line_width = LINE
        print("Setting PIFA...")
        fr4 = ['With Material', '.Reset', '.Name "FR-4 (loss free)"', '.Folder ""', 
               '.FrqType "all"', '.Type "Normal"', '.SetMaterialUnit "GHz", "mm"', 
               '.Epsilon "4.3"', '.Mu "1.0"', '.Kappa "0.0"', '.TanD "0.0"', '.TanDFreq "0.0"', 
               '.TanDGiven "False"', '.TanDModel "ConstTanD"', '.KappaM "0.0"', '.TanDM "0.0"', 
               '.TanDMFreq "0.0"', '.TanDMGiven "False"', '.TanDMModel "ConstKappa"', 
               '.DispModelEps "None"', '.DispModelMu "None"', '.DispersiveFittingSchemeEps "General 1st"', 
               '.DispersiveFittingSchemeMu "General 1st"', '.UseGeneralDispersionEps "False"', 
               '.UseGeneralDispersionMu "False"', '.Rho "0.0"', '.ThermalType "Normal"', 
               '.ThermalConductivity "0.3"', '.SetActiveMaterial "all"', '.Colour "0.75", "0.95", "0.85"', 
               '.Wireframe "False"', '.Transparency "0"', '.Create', 'End With']
        substrate = ['With Brick', '.Reset', '.Name "substrate"', '.Component "component1"', '.Material "FR-4 (loss free)"', 
                     f'.Xrange "0", "75"', f'.Yrange "-{y_offset}", "150-{y_offset}"', 
                     '.Zrange "-1.6", "0"', '.Create', 'End With']
        ground = ['With Brick', '.Reset', '.Name "ground"', '.Component "component1"', 
                  '.Material "PEC"', f'.Xrange "0", "75"', f'.Yrange "-{y_offset}", "0"', 
                  '.Zrange "0", "0.035"', '.Create', 'End With']
        pin = ['With Brick', '.Reset', '.Name "pin"', '.Component "component1"', '.Material "PEC"', 
               f'.Xrange "fx-pin_dis", "fx-pin_dis+{line_width}"', '.Yrange "0", "pin_width"', '.Zrange "0", "0.035"', 
               '.Create', 'End With']
        patch = ['With Brick', '.Reset', '.Name "patch"', '.Component "component1"', '.Material "PEC"', 
                 f'.Xrange "{self.block_len+PATCH_OFFSET}", "patch_len+{self.block_len}"', 
                 f'.Yrange "pin_width", "pin_width+{line_width}"', 
                 '.Zrange "0", "0.035"', '.Create', 'End With']
        feed = ['With Brick', '.Reset', '.Name "feed"', '.Component "component1"', '.Material "PEC"', 
                f'.Xrange "fx-{line_width/2}", "fx+{line_width/2}"', '.Yrange "1", "pin_width"', '.Zrange "0", "0.035"', 
                '.Create', 'End With']
        command = fr4 + substrate + ground + pin + patch + feed
        command = "\n".join(command)
        self.prj.modeler.add_to_history("initialize",command)
        self.save()
        print("PIFA set")
    
    def set_blocks(self): 
        print("Setting blocks...")
        command = []
        binary_sequence = [0 for _ in range(self.blocks_num)]
        self.update_distribution(binary_sequence) # Define materials first
        pos = []
        y_pos = 0
        xleft = -self.block_sep
        xright = self.desRegionX + self.block_len + self.block_sep
        for i in range(self.blocks_num):
            if (i%2) == 0: # left
                if y_pos <= self.desRegionY:pos.append((0, y_pos))
                else:
                    xleft += self.block_len + self.block_sep
                    if xright-xleft < self.block_len:
                        print(f"Too many blocks. Total {i} blocks generated.")
                        break 
                    else: pos.append((xleft, self.desRegionY))
            else: # right
                if y_pos < self.desRegionY:
                    pos.append((self.desRegionX+self.block_len, y_pos))
                    y_pos += self.block_len + self.block_sep
                elif y_pos == self.desRegionY: # y_pos aligned to max y range
                    xleft = 0 # align xleft
                    xright = self.desRegionX + self.block_len # align xright
                    pos.append((self.desRegionX+self.block_len, y_pos))
                    y_pos += self.block_len + self.block_sep
                else:
                    xright -= self.block_len + self.block_sep
                    if xright-xleft < self.block_len:
                        print(f"Too many blocks. Total {i} blocks generated.")
                        break 
                    else: pos.append((xright, self.desRegionY))               
        for key, val in enumerate(pos): command += self.create_blocks(key, val[0], val[0]+self.block_len, val[1], val[1]+self.block_len)
        command = "\n".join(command)
        self.prj.modeler.add_to_history("domain",command)
        self.save()
        print("Blocks set")
        return pos

    def set_port(self):
        command = ['With DiscretePort', '.Reset', '.PortNumber "1"', 
                   '.Type "SParameter"', '.Label ""', '.Folder ""', '.Impedance "50.0"', 
                   '.Voltage "1.0"', '.Current "1.0"', '.Monitor "True"', '.Radius "0.0"', 
                   '.SetP1 "False", "fx", "1", "0.035"', '.SetP2 "False", "fx", "0", "0.035"', 
                '.InvertDirection "False"', '.LocalCoordinates "False"', '.Wire ""',
                '.Position "end1"', '.Create', 'End With']
        # self.excute_vba(command)
        command = "\n".join(command)
        self.prj.modeler.add_to_history("port",command)
        self.save()
        print("Port set")
    
    def delete_port(self):
        command = ['Sub Main', 'Port.Delete "1"', 'End Sub'] #
        self.excute_vba(command)
        print("Port deleted")

    def optimize(self, feedx): # input_seq = [feedx, b0, b1, b2, ..., bn] 
        print("Optimizing...")
        task = ['Sub Main','With SimulationTask', '.Reset', '.Name ("Opt")', 'If Not .DoesExist Then', 
                '.Type ("Optimization")', '.Create', '.Reset', '.Type ("S-Parameters")', 
                '.Name ("Opt\\SPar")', f'.SetProperty("fmin", {FMIN})', f'.SetProperty("fmax", {FMAX})', 
                 '.SetProperty("maximum frequency range", "False")', '.Create', 'End If', 'End With']
        optimizer = ['With Optimizer', '.SetSimulationType( "Opt" )', 
                     '.SetOptimizerType("Trust_Region")', '.SetAlwaysStartFromCurrent(False)', 
                     '.InitParameterList']
        for name, val in self.parameters.items():
            optimizer.append(f'.SelectParameter("{name}", True)')
            optimizer.append(f'.SetParameterInit({val})')
            if name == "pin_dis":
                optimizer.append('.SetParameterMin({:.2f})'.format(0.5))
                optimizer.append('.SetParameterMax({:.2f})'.format(feedx-self.block_len-PATCH_OFFSET)) 
            elif name == "patch_len":
                optimizer.append('.SetParameterMin({:.2f})'.format(feedx-self.block_len+0.5))
                optimizer.append('.SetParameterMax({:.2f})'.format(REGIONX-PATCH_OFFSET)) 
            elif name == "pin_width":
                optimizer.append('.SetParameterMin({:.2f})'.format(LINE))
                optimizer.append('.SetParameterMax({:.2f})'.format(REGIONY-LINE-PATCH_OFFSET)) 
        optimizer.append('.DeleteAllGoals')
        goal = ['Dim gid As Integer', 'gid = .AddGoal("1DC Primary Result")', '.SelectGoal(gid, True)', 
                '.SetGoal1DCResultName(".\\S-Parameters\\S1,1")', '.SetGoalScalarType("magdB20")', 
                '.SetGoalOperator("<")', f'.SetGoalTarget({GOAL})', '.SetGoalRangeType("range")', 
                f'.SetGoalRange({GFMIN},{GFMAX})', '.Start', 'End With', 'End Sub']
        self.excute_vba(task + optimizer + goal)
        print("Optimized")
    
    def read_optimizer_para(self, para_name):
        param_value = self.read(f'1D Results\\Optimizer\\Parameters\\{para_name}') # [(step, value)]
        param_value = param_value[-1][-1]
        print(f"{para_name}= ", param_value)
        return param_value
    
    def update_parameter_dict(self):
        for key in self.parameters.keys(): self.parameters[key] = self.read_optimizer_para(key)

    # Generate input csv file
    def generate_input(self, n_samples=N_SAMPLES, seed=SEED):
        np.random.seed(seed) # set seed for reproducibility
        # fx = 0.5*np.ones((n_samples,1)) # generate feed x position
        fx = np.random.rand(n_samples, 1) # generate feed x position
        fx = fx*(self.fx_max - self.fx_min) + self.fx_min
        np.random.seed(seed) # set seed for reproducibility
        blocks = np.random.randint(0, 2, (n_samples, self.blocks_num)) # generate blocks_num binary input features (0 or 1)
        data = np.concatenate((fx, blocks), axis=1)
        df = pd.DataFrame(data, columns = ['feedx']+[f'region{i+1}' for i in range(self.blocks_num)]) # create a DataFrame
        df.to_csv('data/input.csv', index=False) # save to CSV
        print("Input data generated and saved to 'data/input.csv'")
        return data
    
    def set_frequency_solver(self):
        command = ['ChangeSolverType "HF Frequency Domain"', 
                   f'Solver.FrequencyRange "{FMIN}", "{FMAX}"']
        command = "\n".join(command)
        self.prj.modeler.add_to_history("freq_range",command)
        self.save()
        print("Frequency solver set")
