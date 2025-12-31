import IrModel
import numpy as np
import tkinter as tk
import os
from PIL import ImageTk, Image
import PIL.Image
dir_path = os.path.dirname(os.path.realpath(__file__))



# Callable function to run the fit
def RunModel(objs, Dust_Geometry = 'Differential', HI_ratios = 'Case B', Ices_6micron = False, BootStrap = False, N_bootstrap = 100, useMCMC = False, InitialFit = False, lam_range=[1.5, 28.0], show_progress = True, N_MCMC = 5000, N_BurnIn = 15000, ExtCurve = 'D23ExtCurve', EmCurve = 'D24Emissivity', MIR_CH = 'CHExt_v3', NIR_CH = 'CH_NIR', Fit_NIR_CH = False, NIR_Ice = 'NIR_Ice', NIR_CO2 = 'NIR_CO2', RegStrength = 10000, Cont_Only = False, St_Cont = True, Extend = False, Fit_CO = False, spec_res = 'h'):
    
    z = 0.0
    skip = True
    for i in range(len(objs)):
        objName = objs[i]
        print("")
        print('Fitting ' +objName +'...') 
        print("")



        try:
            lams, flux, flux_err = np.loadtxt(dir_path+'/Data/'+objName+'.txt', unpack=True, usecols=[0,1,2])
        except:
            try:
                lams, flux, flux_err = np.loadtxt(dir_path+'/Data/'+objName+'.dat', unpack=True, usecols=[0,1,2])
            except:
                try:
                    lams, flux, flux_err = np.loadtxt(dir_path+'/Data/'+objName+'.tbl', unpack=True, usecols=[0,1,2], skiprows= 4)
                except:
                    try:
                        lams, flux, flux_err = np.loadtxt(dir_path+'/Data/'+objName+'.txt', unpack=True, usecols=[0,1,2], skiprows= 4)
                    except:
                        try:
                            lams, flux, flux_err = np.loadtxt(dir_path+'/Data/'+objName+'.dat', unpack=True, usecols=[0,1,2], skiprows= 4)
                        except:
                            lams, flux, flux_err = np.loadtxt(dir_path+'/Data/'+objName+'.tbl', unpack=True, usecols=[0,1,2], skiprows= 4)


        flux_err = flux_err[~np.isnan(flux)]#/10.0
        lams = lams[~np.isnan(flux)]
        flux = flux[~np.isnan(flux)]

        lams = lams[flux_err!=0.0]
        flux = flux[flux_err!=0.0]
        flux_err = flux_err[flux_err!=0.0]
        lams = lams[flux_err>0.0]
        flux = flux[flux_err>0.0]
        flux_err = flux_err[flux_err>0.0]


        specdata=[lams, flux, flux_err]

        if (skip==False or useMCMC==False):
            binNo = 0
            if (BootStrap == True):
                binNo = 2
            output = IrModel.RunFit(objName, specdata, z, lam_range, binNo,  useMCMC=False,   ExtType_=Dust_Geometry, Ices_6micron=Ices_6micron, InitialFit=InitialFit, BootStrap=BootStrap, N_bootstrap = N_bootstrap,  HI_ratios = HI_ratios, show_progress=show_progress, N_MCMC = N_MCMC, N_BurnIn = N_BurnIn, ExtCurve = ExtCurve, EmCurve = EmCurve, MIR_CH = MIR_CH, NIR_CH = NIR_CH, Fit_NIR_CH =  Fit_NIR_CH, NIR_Ice = NIR_Ice, NIR_CO2 = NIR_CO2, RegStrength = RegStrength,  Cont_Only = Cont_Only, St_Cont = St_Cont, Extend = Extend,  Fit_CO = Fit_CO, spec_res = spec_res)        
        if (useMCMC==True):        
            binNo = 1
            output = IrModel.RunFit(objName, specdata, z, lam_range, binNo,  useMCMC=useMCMC,ExtType_=Dust_Geometry, Ices_6micron=Ices_6micron, InitialFit=InitialFit, BootStrap=BootStrap, N_bootstrap = N_bootstrap,  HI_ratios = HI_ratios, show_progress=show_progress, N_MCMC = N_MCMC, N_BurnIn = N_BurnIn, ExtCurve = ExtCurve, EmCurve = EmCurve, MIR_CH = MIR_CH, NIR_CH = NIR_CH, Fit_NIR_CH =  Fit_NIR_CH, NIR_Ice = NIR_Ice, NIR_CO2 = NIR_CO2, RegStrength = RegStrength, Cont_Only = Cont_Only, St_Cont = St_Cont, Extend = Extend, Fit_CO = Fit_CO, spec_res = spec_res)        


###############

from tkinter import *
from tkinter import ttk

class ToolTip(object):

    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        "Display text in tooltip window"
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 57
        y = y + cy + self.widget.winfo_rooty() +27
        self.tipwindow = tw = Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = ttk.Label(tw, text=self.text, justify=LEFT,
                      background="#ffffe0", relief=SOLID, borderwidth=1,
                      font=("tahoma", "12", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

def CreateToolTip(widget, text):
    toolTip = ToolTip(widget)
    def enter(event):
        toolTip.showtip(text)
    def leave(event):
        toolTip.hidetip()
    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)





if __name__ == '__main__':

    import glob, os
    os.chdir("./Data")
    options = [""]
    for file in glob.glob("*"):
        options.append(file.rsplit('.',1)[0])
    options = sorted(options)
    root = Tk()
    root.title("SPIRIT - SPectral InfraRed Inference Tool")
    root.geometry("700x700")  # Increased height to accommodate logo

    # Set custom icon for taskbar
    try:
        root.iconphoto(True, tk.PhotoImage(file=dir_path + "/Icon.png"))
    except:
        pass  # If icon file not found, just use default
    

    
    # Make window pop up and grab focus
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    root.focus_force()

    # Configure grid weights
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=0)  # Logo row - no expansion
    root.rowconfigure(1, weight=1)  # Main content row - expands
    
    # Add logo at the very top of root, not in main_frame
    try:
        logo_image = PIL.Image.open(dir_path + "/Logo.jpg")
        # Resize logo to fit nicely (adjust dimensions as needed)
        logo_image = logo_image.resize((400, 150), PIL.Image.LANCZOS)
        logo_photo = ImageTk.PhotoImage(logo_image)
        logo_label = tk.Label(root, image=logo_photo)
        logo_label.image = logo_photo  # Keep a reference to prevent garbage collection
        logo_label.grid(row=0, column=0, pady=(10, 10), sticky='n')
    except Exception as e:
        # If logo can't be loaded, show text instead
        tk.Label(root, text="SPIRIT - SPectral InfraRed Inference Tool", 
                 font=('Helvetica', 16, 'bold')).grid(row=0, column=0, pady=(10, 10), sticky='n')

    # Create main container BELOW the logo
    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Configure main_frame grid weights
    main_frame.columnconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)
    main_frame.columnconfigure(2, weight=1)
    
    # Left panel - Basic options (back to row 0 in main_frame)
    left_panel = ttk.LabelFrame(main_frame, text="Basic Options", padding="10")
    left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
    
    l1 = ttk.Label(left_panel, text="Spectrum")
    l1.grid(column=0, row=0, sticky=tk.W, pady=5)
    obj_= tk.StringVar()
    dropdown = ttk.OptionMenu(left_panel, obj_, *options)
    dropdown.grid(column=1, row=0, pady=5, sticky=(tk.W, tk.E))
    CreateToolTip(l1, text='Select the name of the spectrum from\nthe Data folder to be fitted.')
    CreateToolTip(dropdown, text='Select the name of the spectrum from\nthe Data folder to be fitted.')

    l2 = ttk.Label(left_panel, text="Dust Geometry")
    l2.grid(column=0, row=1, sticky=tk.W, pady=5)
    ExtType_= tk.StringVar()
    options2 = ["Differential", "Differential", "Screen", "Mixed"]
    dropdown2 = ttk.OptionMenu(left_panel, ExtType_, *options2)
    dropdown2.grid(column=1, row=1, pady=5, sticky=(tk.W, tk.E))
    CreateToolTip(l2, text='Select which dust geometry to use.')
    CreateToolTip(dropdown2, text='Select which dust geometry to use.')

    l3 = ttk.Label(left_panel, text="HI Ratios")
    l3.grid(column=0, row=2, sticky=tk.W, pady=5)
    HI_ratios_= tk.StringVar()
    options3 = ["Case B", "Case B", "CLOUDY"]
    dropdown3 = ttk.OptionMenu(left_panel, HI_ratios_, *options3)
    dropdown3.grid(column=1, row=2, pady=5, sticky=(tk.W, tk.E))
    CreateToolTip(l3, text='Choose whether to use Case B or CLOUDY to define the\nintrinsic HI ratios when inferring the HI extinction. This option does not affect the best fit model, only the derived extinction values.')
    CreateToolTip(dropdown3, text='Choose whether to use Case B or CLOUDY to define the\nintrinsic HI ratios when inferring the HI extinction. This option does not affect the best fit model, only the derived extinction values.')

    l4 = ttk.Label(left_panel, text="6 Micron Ices?")
    l4.grid(column=0, row=3, sticky=tk.W, pady=5)
    CheckVar1 = IntVar()
    C1 = ttk.Checkbutton(left_panel, variable=CheckVar1, onvalue=True, offvalue=False)
    C1.grid(column=1, row=3, pady=5, sticky=tk.W)
    CreateToolTip(l4, text='Whether to fit the 6 micron ice feature.')
    CreateToolTip(C1, text='Whether to fit the 6 micron ice feature.')

    l6 = ttk.Label(left_panel, text="Show Progress")
    l6.grid(column=0, row=4, sticky=tk.W, pady=5)
    CheckVar2 = IntVar(value=True)
    C2 = ttk.Checkbutton(left_panel, variable=CheckVar2, onvalue=True, offvalue=False)
    C2.grid(column=1, row=4, pady=5, sticky=tk.W)
    CreateToolTip(l6, text='Whether to display the progress of the fitting.')
    CreateToolTip(C2, text='Whether to display the progress of the fitting.')
    
    left_panel.columnconfigure(1, weight=1)
    
    # Middle panel - Fitting options (back to row 0 in main_frame)
    middle_panel = ttk.LabelFrame(main_frame, text="Fitting Options", padding="10")
    middle_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
    
    l5 = ttk.Label(middle_panel, text="Fitting Method")
    l5.grid(column=0, row=0, sticky=tk.W, pady=5)
    FitMethod_= tk.StringVar()
    options_fit = ["Quick", "Quick", "BootStrap", "MCMC"]
    dropdown_fit = ttk.OptionMenu(middle_panel, FitMethod_, *options_fit)
    dropdown_fit.grid(column=1, row=0, pady=5, sticky=(tk.W, tk.E))
    CreateToolTip(l5, text='Choose the fitting method.\n'
                            'Quick will find the max probability fitting model. This is the quickest method but will provide no uncertanties.\n'
                            'BootStrap, will repeat the quick fit many times, resampling the data each time. This method therefore provides uncertanties.\n'
                            'MCMC uses NumPyro NUTS to find the best fit, providing uncertanties.')
    CreateToolTip(dropdown_fit, text='Choose the fitting method.\n'
                            'Quick will find the max probability fitting model. This is the quickest method but will provide no uncertanties.\n'
                            'BootStrap, will repeat the quick fit many times, resampling the data each time. This method therefore provides uncertanties.\n'
                            'MCMC uses NumPyro NUTS to find the best fit, providing uncertanties.')

    BootStrap_label = ttk.Label(middle_panel, text="Number of BootStraps")
    CreateToolTip(BootStrap_label, text='Select the number of time to resample/refit the data.')
    NBootstrap_= tk.IntVar()
    options4 = [100, 10, 50, 100, 500, 1000]
    dropdown4 = ttk.OptionMenu(middle_panel, NBootstrap_, *options4)
    CreateToolTip(dropdown4, text='Select the number of time to resample/refit the data.')
 
    MCMC_label = ttk.Label(middle_panel, text="MCMC Samples")
    CreateToolTip(MCMC_label, text='The number of MCMC samples after the burn-in is discared.')
    NSamples_= tk.IntVar()
    options5 = [5000, 1000, 2000, 3000, 4000, 5000, 10000]
    dropdown5 = ttk.OptionMenu(middle_panel, NSamples_, *options5)
    CreateToolTip(dropdown5, text='The number of MCMC samples after the burn-in is discared.')

    MCMC_label2 = ttk.Label(middle_panel, text="MCMC Burn In")
    CreateToolTip(MCMC_label2, text='The number of burn-in MCMC samples.')
    NBurnIn_= tk.IntVar()
    options6 = [15000, 5000, 10000, 15000, 20000, 25000, 30000]
    dropdown6 = ttk.OptionMenu(middle_panel, NBurnIn_, *options6)
    CreateToolTip(dropdown6, text='The number of burn-in MCMC samples.')

    MCMC_label3 = ttk.Label(middle_panel, text="Initial Quick Fit")
    CreateToolTip(MCMC_label3, text='Whether to run an initial fit with the Quick method to help initialise the MCMC samples.\n'
                                        'This may enable a shorter burn-in size.')
    CheckVar4 = IntVar()
    C4 = ttk.Checkbutton(middle_panel, variable=CheckVar4, onvalue=True, offvalue=False)
    CreateToolTip(C4, text='Whether to run an initial fit with the Quick method to help initialise the MCMC samples.\n'
                                        'This may enable a shorter burn-in size.')

    def options_callback(*args):
        # Clear all first
        for widget in [BootStrap_label, dropdown4, MCMC_label, dropdown5, MCMC_label2, dropdown6, MCMC_label3, C4]:
            widget.grid_remove()
        
        if (FitMethod_.get() == 'BootStrap'):
            BootStrap_label.grid(column=0, row=1, sticky=tk.W, pady=5)
            dropdown4.grid(column=1, row=1, pady=5, sticky=(tk.W, tk.E))
        elif (FitMethod_.get() == 'MCMC'):
            MCMC_label.grid(column=0, row=1, sticky=tk.W, pady=5)
            dropdown5.grid(column=1, row=1, pady=5, sticky=(tk.W, tk.E))
            MCMC_label2.grid(column=0, row=2, sticky=tk.W, pady=5)
            dropdown6.grid(column=1, row=2, pady=5, sticky=(tk.W, tk.E))
            MCMC_label3.grid(column=0, row=3, sticky=tk.W, pady=5)
            C4.grid(column=1, row=3, pady=5, sticky=tk.W)

    FitMethod_.trace("w", options_callback)
    middle_panel.columnconfigure(1, weight=1)
    
    # Right panel - Advanced options toggle (back to row 0 in main_frame)
    right_panel = ttk.LabelFrame(main_frame, text="Advanced Options", padding="10")
    right_panel.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
    
    adv_label = ttk.Label(right_panel, text="Show Advanced Options")
    adv_label.grid(column=0, row=0, sticky=tk.W, pady=5)
    CheckVar3 = IntVar()
    C3 = ttk.Checkbutton(right_panel, variable=CheckVar3, onvalue=True, offvalue=False)
    C3.grid(column=1, row=0, pady=5, sticky=tk.W)
    CreateToolTip(adv_label, text='Show or hide advanced options')
    CreateToolTip(C3, text='Show or hide advanced options')
    
    right_panel.columnconfigure(1, weight=1)
    
    # Bottom panel - Advanced options content (back to row 1 in main_frame)
    advanced_panel = ttk.Frame(main_frame, padding="10")
    advanced_panel.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
    
    os.chdir(dir_path +"/Ext.Curves")
    options7 = ["D23ExtCurve"]
    for file in glob.glob("*"):
        options7.append(file.rsplit('.',1)[0])

    ExtCurve_label = ttk.Label(advanced_panel, text="Extinction Curve")
    CreateToolTip(ExtCurve_label, text='Select which extinction curve to use from the Ext.Curves folder.')
    ExtCurve_= tk.StringVar()
    dropdown7 = ttk.OptionMenu(advanced_panel, ExtCurve_, *options7)
    CreateToolTip(dropdown7, text='Select which extinction curve to use from the Ext.Curves folder.')

    os.chdir(dir_path +"/Emissivity")
    options8 = ["D24Emissivity"]
    for file in glob.glob("*"):
        options8.append(file.rsplit('.',1)[0])

    EmCurve_label = ttk.Label(advanced_panel, text="Emissivity Curve")
    CreateToolTip(EmCurve_label, text='Select which emissivity curve to use from the Emissivity folder.')
    EmCurve_= tk.StringVar()
    dropdown8 = ttk.OptionMenu(advanced_panel, EmCurve_, *options8)
    CreateToolTip(dropdown8, text='Select which emissivity curve to use from the Emissivity folder.')

    os.chdir(dir_path +"/IceTemplates/MIR_CH")
    options9 = ["CHExt_v3"]
    for file in glob.glob("*"):
        options9.append(file.rsplit('.',1)[0])
        
    l7 = ttk.Label(advanced_panel, text="MIR CH Template")
    CreateToolTip(l7, text='Select the CH template for the ~7.6 micron absorption features.\n'
                            'This is only fitted if the 6 micron ices are enabled.')
    MIR_CH_= tk.StringVar()
    dropdown9 = ttk.OptionMenu(advanced_panel, MIR_CH_, *options9)
    CreateToolTip(dropdown9, text='Select the CH template for the ~7.6 micron absorption features.\n'
                            'This is only fitted if the 6 micron ices are enabled.')

    os.chdir(dir_path +"/IceTemplates/NIR_CH")
    options10 = ["CH_NIR"]
    for file in glob.glob("*"):
        options10.append(file.rsplit('.',1)[0])
        
    l8 = ttk.Label(advanced_panel, text="NIR CH Template")
    CreateToolTip(l8, text='Select the CH template for the 3.4 micron absorption feature. This feature is only fitted if "Fit NIR CH?" is enabled. ')
    NIR_CH_= tk.StringVar()
    dropdown10 = ttk.OptionMenu(advanced_panel, NIR_CH_, *options10)
    CreateToolTip(dropdown10, text='Select the CH template for the 3.4 micron absorption feature. This feature is only fitted if "Fit NIR CH?" is enabled. ')

    os.chdir(dir_path +"/IceTemplates/NIR_CO2")
    options11 = ["NIR_CO2"]
    for file in glob.glob("*"):
        options11.append(file.rsplit('.',1)[0])
        
    l9 = ttk.Label(advanced_panel, text="NIR CO2 Template")
    CreateToolTip(l9, text='Select the CO2 template for the 4.25 micron absorption feature.')
    NIR_CO2_= tk.StringVar()
    dropdown11 = ttk.OptionMenu(advanced_panel, NIR_CO2_, *options11)
    CreateToolTip(dropdown11, text='Select the CO2 template for the 4.25 micron absorption feature.')

    os.chdir(dir_path +"/IceTemplates/NIR_H2O")
    options12 = ["NIR_Ice"]
    for file in glob.glob("*"):
        options12.append(file.rsplit('.',1)[0])
        
    l10 = ttk.Label(advanced_panel, text="NIR H2O Template")
    CreateToolTip(l10, text='Select the ice template for the 3 micron absorption feature.')
    NIR_Ice_= tk.StringVar()
    dropdown12 = ttk.OptionMenu(advanced_panel, NIR_Ice_, *options12)
    CreateToolTip(dropdown12, text='Select the ice template for the 3 micron absorption feature.')

    l13 = ttk.Label(advanced_panel, text="Fit NIR CH?")
    Fit_NIR_CH = IntVar()
    C4_adv = ttk.Checkbutton(advanced_panel, variable=Fit_NIR_CH, onvalue=True, offvalue=False)
    CreateToolTip(l13, text='Whether to fit the 3.4 micron CH absorption feature.')
    CreateToolTip(C4_adv, text='Whether to fit the 3.4 micron CH absorption feature.')

    l14 = ttk.Label(advanced_panel, text="Regularisation Strength")
    RegStrength_ = IntVar()
    options13 = [10000, 10, 100, 1000, 10000, 100000]
    dropdown13 = ttk.OptionMenu(advanced_panel, RegStrength_, *options13)
    CreateToolTip(l14, text='Strength of regularisation on the shape of the dust distribution.\n'
        'This is the factor Gamma in equation 4 of Donnan+24.')
    CreateToolTip(dropdown13, text='Strength of regularisation on the shape of the dust distribution.\n'
        'This is the factor Gamma in equation 4 of Donnan+24.')

    l15 = ttk.Label(advanced_panel, text="Continuum Only")
    Cont_Only = IntVar()
    C5 = ttk.Checkbutton(advanced_panel, variable=Cont_Only, onvalue=True, offvalue=False)
    CreateToolTip(l15, text='Disable the PAH features and fit only the continuum.')
    CreateToolTip(C5, text='Disable the PAH features and fit only the continuum.')

    l16 = ttk.Label(advanced_panel, text="Disable Stellar Continuum")
    St_Cont = IntVar()
    C6 = ttk.Checkbutton(advanced_panel, variable=St_Cont, onvalue=True, offvalue=False)
    CreateToolTip(l16, text='Disable the stellar continuum.')
    CreateToolTip(C6, text='Disable the stellar continuum.')

    l17 = ttk.Label(advanced_panel, text="Extend Dust Distribtion")
    Extend = IntVar()
    C7 = ttk.Checkbutton(advanced_panel, variable=Extend, onvalue=True, offvalue=False)
    CreateToolTip(l17, text='Extend the dust distribution grid for the differential\nextinction model, to include even higher extinctions.')
    CreateToolTip(C7, text='Extend the dust distribution grid for the differential\nextinction model, to include even higher extinctions.')
    
    l18 = ttk.Label(advanced_panel, text="Fit Ro.Vib. CO")
    Fit_CO = IntVar()
    C8 = ttk.Checkbutton(advanced_panel, variable=Fit_CO, onvalue=True, offvalue=False)
    CreateToolTip(l18, text='Include broad CO absoprtion at ~4.6 micron to account for CO ro-vibrational + ice band. (Useful for AGN/Obscured nuclei) )')
    CreateToolTip(C8, text='Include broad CO absoprtion at ~4.6 micron to account for CO ro-vibrational + ice band. (Useful for AGN/Obscured nuclei)')

    def options_callback2(*args):
        if (CheckVar3.get() == True):
            # Grid in 4 columns for compact layout
            ExtCurve_label.grid(column=0, row=0, sticky=tk.W, pady=5, padx=5)
            dropdown7.grid(column=1, row=0, pady=5, padx=5, sticky=(tk.W, tk.E))
            
            EmCurve_label.grid(column=2, row=0, sticky=tk.W, pady=5, padx=5)
            dropdown8.grid(column=3, row=0, pady=5, padx=5, sticky=(tk.W, tk.E))
            
            l7.grid(column=0, row=1, sticky=tk.W, pady=5, padx=5)
            dropdown9.grid(column=1, row=1, pady=5, padx=5, sticky=(tk.W, tk.E))
            
            l8.grid(column=2, row=1, sticky=tk.W, pady=5, padx=5)
            dropdown10.grid(column=3, row=1, pady=5, padx=5, sticky=(tk.W, tk.E))
            
            l9.grid(column=0, row=2, sticky=tk.W, pady=5, padx=5)
            dropdown11.grid(column=1, row=2, pady=5, padx=5, sticky=(tk.W, tk.E))
            
            l10.grid(column=2, row=2, sticky=tk.W, pady=5, padx=5)
            dropdown12.grid(column=3, row=2, pady=5, padx=5, sticky=(tk.W, tk.E))
            
            l13.grid(column=0, row=3, sticky=tk.W, pady=5, padx=5)
            C4_adv.grid(column=1, row=3, pady=5, sticky=tk.W)
            
            l14.grid(column=2, row=3, sticky=tk.W, pady=5, padx=5)
            dropdown13.grid(column=3, row=3, pady=5, padx=5, sticky=(tk.W, tk.E))
            
            l15.grid(column=0, row=4, sticky=tk.W, pady=5, padx=5)
            C5.grid(column=1, row=4, pady=5, sticky=tk.W)
            
            l16.grid(column=2, row=4, sticky=tk.W, pady=5, padx=5)
            C6.grid(column=3, row=4, pady=5, sticky=tk.W)
            
            l17.grid(column=0, row=5, sticky=tk.W, pady=5, padx=5)
            C7.grid(column=1, row=5, pady=5, sticky=tk.W)
            
            l18.grid(column=2, row=5, sticky=tk.W, pady=5, padx=5)
            C8.grid(column=3, row=5, pady=5, sticky=tk.W)
        else:
            for widget in [ExtCurve_label, dropdown7, EmCurve_label, dropdown8, l7, dropdown9, 
                          l8, dropdown10, l9, dropdown11, l10, dropdown12, l13, C4_adv, 
                          l14, dropdown13, l15, C5, l16, C6, l17, C7, l18, C8]:
                widget.grid_remove()

    CheckVar3.trace("w", options_callback2)
    
    for i in range(4):
        advanced_panel.columnconfigure(i, weight=1)
    
    # Button panel at bottom (back to row 2 in main_frame)
    button_panel = ttk.Frame(main_frame)
    button_panel.grid(row=2, column=0, columnspan=3, pady=10)
    
    run = ttk.Button(button_panel, text="Run", command=root.destroy)
    run.grid(column=0, row=0, padx=5)
    
    root.mainloop()

    print("")
    obj = obj_.get()
    ExtType = ExtType_.get()
    HI_ratios = HI_ratios_

    FitMethod = FitMethod_.get()
    #Defaults
    NBootstrap = 100
    N_MCMC = 5000
    N_BurnIn = 15000
    if (FitMethod == 'Quick'):
        BootStrap = False
        useMCMC = False
    elif (FitMethod == 'BootStrap'):
        BootStrap = True
        useMCMC = False
        NBootstrap = NBootstrap_.get()
    else:
        BootStrap = False
        useMCMC = True
        N_MCMC = NSamples_.get()
        N_BurnIn = NBurnIn_.get()

    #if ():
    ExtCurve = ExtCurve_.get()
    EmCurve = EmCurve_.get()
    # else:
    #     ExtCurve = 'D23ExtCurve'

    MIR_CH = MIR_CH_.get()
    NIR_CH = NIR_CH_.get()
    NIR_Ice = NIR_Ice_.get()
    NIR_CO2 = NIR_CO2_.get()
    Fit_NIR_CH = Fit_NIR_CH.get()

    RegStrength = RegStrength_.get()

    if (obj == ""):
        print('Exiting...')
    else:
        RunModel([obj], Dust_Geometry = ExtType, HI_ratios = HI_ratios, Ices_6micron = CheckVar1.get(), BootStrap = BootStrap, N_bootstrap = NBootstrap, useMCMC = useMCMC, InitialFit = CheckVar4.get(), lam_range=[1.5, 28.0], show_progress = CheckVar2.get(), N_MCMC = N_MCMC, N_BurnIn = N_BurnIn, ExtCurve = ExtCurve, EmCurve=EmCurve, MIR_CH = MIR_CH, NIR_CH = NIR_CH, Fit_NIR_CH = Fit_NIR_CH, NIR_Ice = NIR_Ice, NIR_CO2 = NIR_CO2, RegStrength=RegStrength, Cont_Only = Cont_Only.get(), St_Cont = St_Cont.get(), Extend = Extend.get(), Fit_CO = Fit_CO.get())


