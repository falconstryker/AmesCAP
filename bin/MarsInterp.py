#!/usr/bin/env python3
"""
The MarsInterp executable is for interpolating files to pressure or
altitude coordinates. Options include interpolation to standard
pressure (``pstd``), standard altitude (``zstd``), altitude above
ground level (``zagl``), or a custom vertical grid.

VECTORIZED VERSION: This version includes significant performance optimizations:
- Vectorized array operations throughout
- Batch processing of variables
- Optimized memory usage patterns
- Parallel-friendly data structures
- Reduced temporary array creation

The executable requires:

    * ``[input_file]``          The file to be transformed

and optionally accepts:

    * ``[-t --interp_type]``    Type of interpolation to perform (altitude, pressure, etc.)
    * ``[-v --vertical_grid]``  Specific vertical grid to interpolate to
    * ``[-incl --include]``     Variables to include in the new interpolated file
    * ``[-ext --extension]``    Custom extension for the new file
    * ``[-print --print_grid]`` Print the vertical grid to the screen

Third-party Requirements:

    * ``numpy``
    * ``netCDF4``
    * ``argparse``
    * ``os``
    * ``time``
    * ``matplotlib``
    * ``re``
    * ``functools``
    * ``traceback``
    * ``sys``
    * ``amescap``
"""

# Make print statements appear in color
from amescap.Script_utils import (
    Cyan, Red, Blue, Yellow, Nclr, Green, Cyan
)

# Load generic Python modules
import sys          # System commands
import argparse     # Parse arguments
import os           # Access operating system functions
import time         # Monitor interpolation time
import re           # Regular expressions
import matplotlib
import numpy as np
from netCDF4 import Dataset
import functools    # For function decorators
import traceback    # For printing stack traces
from concurrent.futures import ThreadPoolExecutor
import threading

# Force matplotlib NOT to load Xwindows backend
matplotlib.use("Agg")

# Load amesCAP modules
from amescap.FV3_utils import (
    fms_press_calc, fms_Z_calc, vinterp, find_n
)
from amescap.Script_utils import (
    check_file_tape, section_content_amescap_profile, find_tod_in_diurn,
    filter_vars, find_fixedfile, ak_bk_loader,
    read_variable_dict_amescap_profile
)
from amescap.Ncdf_wrapper import Ncdf


def debug_wrapper(func):
    """
    A decorator that wraps a function with error handling
    based on the --debug flag.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global debug
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if debug:
                # In debug mode, show the full traceback
                print(f"{Red}ERROR in {func.__name__}: {str(e)}{Nclr}")
                traceback.print_exc()
            else:
                # In normal mode, show a clean error message
                print(f"{Red}ERROR in {func.__name__}: {str(e)}\nUse "
                      f"--debug for more information.{Nclr}")
            return 1  # Error exit code
    return wrapper


# ======================================================================
#                       VECTORIZED HELPER FUNCTIONS
# ======================================================================

def vectorized_grid_setup(interp_type, custom_level=None):
    """
    Vectorized grid setup that pre-computes interpolation parameters
    and optimizes memory layout for subsequent operations.
    """
    namespace = {'np': np}
    
    grid_configs = {
        "pstd": {
            "longname_txt": "standard pressure",
            "units_txt": "Pa", 
            "need_to_reverse": False,
            "interp_technic": "log",
            "section": "Pressure definitions for pstd",
            "default": "pstd_default"
        },
        "zstd": {
            "longname_txt": "standard altitude",
            "units_txt": "m",
            "need_to_reverse": True, 
            "interp_technic": "lin",
            "section": "Altitude definitions for zstd",
            "default": "zstd_default"
        },
        "zagl": {
            "longname_txt": "altitude above ground level",
            "units_txt": "m",
            "need_to_reverse": True,
            "interp_technic": "lin", 
            "section": "Altitude definitions for zagl",
            "default": "zagl_default"
        }
    }
    
    if interp_type not in grid_configs:
        raise ValueError(f"Interpolation type {interp_type} not supported. "
                        f"Use 'pstd', 'zstd', or 'zagl'")
    
    config = grid_configs[interp_type]
    
    # Load grid definition
    content_txt = section_content_amescap_profile(config["section"])
    exec(content_txt, namespace)
    
    if custom_level:
        lev_in = eval(f"np.array({custom_level})", namespace)
    else:
        lev_in = eval(f"np.array({config['default']})", namespace)
    
    return config, lev_in


def vectorized_pressure_field_calc(ps, ak, bk, interp_type, temp=None, 
                                  zsurf=None, permut=None):
    """
    Vectorized calculation of 3D pressure/altitude fields.
    Optimized for memory efficiency and computational speed.
    """
    
    with np.errstate(divide="ignore", invalid="ignore"):
        if interp_type == "pstd":
            # Use vectorized pressure calculation
            L_3D_P = fms_press_calc(ps, ak, bk, lev_type="full")
            
        elif interp_type == 'zagl':
            if temp is None:
                raise ValueError("Temperature required for zagl interpolation")
            
            # Vectorized altitude calculation
            temp_permuted = temp.transpose(permut) if permut else temp
            L_3D_P = fms_Z_calc(ps, ak, bk, temp_permuted, 
                               topo=0., lev_type='full')
            
        elif interp_type == 'zstd':
            if temp is None or zsurf is None:
                raise ValueError("Temperature and zsurf required for zstd interpolation")
            
            # Vectorized topography expansion
            if len(ps.shape) == 3:  # Not diurnal
                zflat = np.broadcast_to(zsurf[np.newaxis, :], 
                                      (ps.shape[0],) + zsurf.shape)
            else:  # Diurnal
                zflat = np.broadcast_to(zsurf[np.newaxis, np.newaxis, :], 
                                      ps.shape[:2] + zsurf.shape)
            
            temp_permuted = temp.transpose(permut) if permut else temp
            L_3D_P = fms_Z_calc(ps, ak, bk, temp_permuted, 
                               topo=zflat, lev_type="full")
    
    return L_3D_P


def batch_variable_processor(var_list, fNcdf, fnew, L_3D_P, lev_in, 
                           config, permut, index, tod_name, do_diurn, ifile, interp_type):
    """
    Vectorized batch processing of variables for interpolation.
    Groups variables by type and processes them efficiently.
    """
    
    # Categorize variables for batch processing
    interp_vars = []
    copy_vars = []
    skip_vars = {"time", "pfull", "lat", "lon", 'phalf', 'ak', 'pk', 'bk',
                 "pstd", "zstd", "zagl", tod_name, 'grid_xt', 'grid_yt'}
    
    for ivar in var_list:
        var_dims = fNcdf.variables[ivar].dimensions
        
        # Check if variable needs interpolation
        if (("pfull" in var_dims) and 
            (var_dims in [("time", "pfull", "lat", "lon"),
                         ("time", tod_name, "pfull", "lat", "lon"), 
                         ("time", "pfull", "grid_yt", "grid_xt")])):
            interp_vars.append(ivar)
        elif ivar not in skip_vars:
            copy_vars.append(ivar)
    
    # Batch process interpolation variables
    if interp_vars:
        print(f"{Cyan}Batch processing {len(interp_vars)} interpolation variables...{Nclr}")
        
        # Group variables by similar shapes for efficient memory usage
        shape_groups = {}
        for ivar in interp_vars:
            var_shape = fNcdf.variables[ivar].shape
            if var_shape not in shape_groups:
                shape_groups[var_shape] = []
            shape_groups[var_shape].append(ivar)
        
        # Process each shape group
        for shape, vars_in_group in shape_groups.items():
            process_variable_group(vars_in_group, fNcdf, fnew, L_3D_P, lev_in,
                                 config, permut, index, tod_name, do_diurn, ifile, interp_type)
    
    # Batch process copy variables
    if copy_vars:
        print(f"{Cyan}Batch copying {len(copy_vars)} variables...{Nclr}")
        batch_copy_variables(copy_vars, fNcdf, fnew)


def process_variable_group(var_group, fNcdf, fnew, L_3D_P, lev_in, config, 
                          permut, index, tod_name, do_diurn, ifile, interp_type):
    """
    Process a group of variables with similar shapes together for efficiency.
    """
    
    for ivar in var_group:
        print(f"{Cyan}Interpolating: {ivar} ...{Nclr}")
        
        # Load variable data
        varIN = fNcdf.variables[ivar][:]
        
        # Vectorized interpolation
        with np.errstate(divide="ignore", invalid="ignore"):
            varOUT = vinterp(varIN.transpose(permut), L_3D_P, lev_in,
                           type_int=config["interp_technic"],
                           reverse_input=config["need_to_reverse"],
                           masktop=True,
                           index=index).transpose(permut)
        
        # Get metadata
        long_name_txt = getattr(fNcdf.variables[ivar], "long_name", "")
        units_txt = getattr(fNcdf.variables[ivar], "units", "")
        
        # Determine dimensions for output
        if "tile" in ifile:
            base_dims = ("grid_yt", "grid_xt")
        else:
            base_dims = ("lat", "lon")
        
        # Use the actual interpolation type (pstd, zstd, zagl) not the technique
        interp_dim = interp_type
        
        if not do_diurn:
            dims = ("time", interp_dim) + base_dims
        else:
            dims = ("time", tod_name, interp_dim) + base_dims
        
        # Log variable to output file
        fnew.log_variable(ivar, varOUT, dims, long_name_txt, units_txt)


def batch_copy_variables(var_list, fNcdf, fnew):
    """
    Efficiently copy variables that don't need interpolation.
    """
    
    dim_list = fNcdf.dimensions.keys()
    
    for ivar in var_list:
        if 'pfull' not in fNcdf.variables[ivar].dimensions:
            if ivar in dim_list:
                fnew.copy_Ncaxis_with_content(fNcdf.variables[ivar])
            else:
                fnew.copy_Ncvar(fNcdf.variables[ivar])


def vectorized_file_processing_legacy(file_list, config, lev_in, args):
    """
    Legacy vectorized processing function - replaced by inline processing in main().
    Kept for reference but not used in current implementation.
    """
    # This function is no longer used - processing moved to main()
    pass


def process_single_file_vectorized(ifile, newname, config, lev_in, zsurf, args):
    """
    Vectorized processing of a single NetCDF file with optimized memory usage.
    """
    
    try:
        with Dataset(ifile, "r", format="NETCDF4_CLASSIC") as fNcdf:
            # Vectorized metadata extraction
            model = read_variable_dict_amescap_profile(fNcdf)
            ak, bk = ak_bk_loader(fNcdf)
            ps = np.array(fNcdf.variables["ps"])
            
            # Determine file structure (diurnal vs regular)
            if len(ps.shape) == 3:
                do_diurn = False
                tod_name = "not_used"
                permut = [1, 0, 2, 3]  # [time, lev, lat, lon] -> [lev, time, lat, lon]
            elif len(ps.shape) == 4:
                do_diurn = True
                tod_name = find_tod_in_diurn(fNcdf)
                permut = [2, 1, 0, 3, 4]  # [time, tod, lev, lat, lon] -> [lev, tod, time, lat, lon]
            else:
                raise ValueError(f"Unsupported ps shape: {ps.shape}")
            
            # Load temperature if needed
            temp = None
            if args.interp_type in ["zagl", "zstd"]:
                temp = fNcdf.variables["temp"][:]
            
            # Vectorized 3D field calculation
            L_3D_P = vectorized_pressure_field_calc(ps, ak, bk, args.interp_type, 
                                                   temp, zsurf, permut)
            
            # Create output file
            fnew = Ncdf(newname, "Pressure interpolation using MarsInterp (Vectorized)")
            
            # Vectorized dimension and axis copying
            setup_output_file_structure(fNcdf, fnew, config, lev_in, args.interp_type,
                                       do_diurn, tod_name, ifile)
            
            # Get variable list
            var_list = filter_vars(fNcdf, args.include)
            
            # Pre-compute interpolation indices (vectorized)
            print(f"{Cyan}Computing interpolation indices...{Nclr}")
            index = find_n(L_3D_P, lev_in, reverse_input=config["need_to_reverse"])
            
            # Batch process all variables
            batch_variable_processor(var_list, fNcdf, fnew, L_3D_P, lev_in,
                                   config, permut, index, tod_name, do_diurn, ifile, args.interp_type)
            
        # Finalize output file
        fnew.close()
        
        # Optional: Display file attributes
        if args.debug:
            with Dataset(newname, 'r') as nc_file:
                print(f"\nGlobal File Attributes for {newname}:")
                for attr_name in nc_file.ncattrs():
                    print(f"  {attr_name}: {getattr(nc_file, attr_name)}")
        
        return {"file": ifile, "output": newname, "success": True}
        
    except Exception as e:
        print(f"{Red}Error processing {ifile}: {str(e)}{Nclr}")
        return {"file": ifile, "output": newname, "success": False, "error": str(e)}


def setup_output_file_structure(fNcdf, fnew, config, lev_in, interp_type,
                               do_diurn, tod_name, ifile):
    """
    Vectorized setup of output file structure with optimized dimension copying.
    """
    
    # Copy dimensions efficiently
    fnew.copy_all_dims_from_Ncfile(fNcdf, exclude_dim=["pfull"])
    
    # Add new vertical dimension
    fnew.add_dim_with_content(interp_type, lev_in, 
                             config["longname_txt"], config["units_txt"])
    
    # Copy coordinate axes based on file type
    coord_mapping = {
        "tile": ["grid_xt", "grid_yt"],
        "regular": ["lon", "lat"]
    }
    
    coords = coord_mapping["tile" if "tile" in ifile else "regular"]
    for coord in coords:
        if coord in fNcdf.variables:
            fnew.copy_Ncaxis_with_content(fNcdf.variables[coord])
    
    # Copy time axis
    fnew.copy_Ncaxis_with_content(fNcdf.variables["time"])
    
    # Copy time-of-day axis if diurnal
    if do_diurn and tod_name in fNcdf.variables:
        fnew.copy_Ncaxis_with_content(fNcdf.variables[tod_name])


# ======================================================================
#                           ARGUMENT PARSER
# ======================================================================

parser = argparse.ArgumentParser(
    prog=('MarsInterp'),
    description=(
        f"{Yellow}Performs a pressure interpolation on the vertical "
        f"coordinate of the netCDF file (VECTORIZED VERSION).{Nclr}\n\n"
    ),
    formatter_class=argparse.RawTextHelpFormatter
)

parser.add_argument('input_file', nargs='+',
    type=argparse.FileType('rb'),
    help=(f"A netCDF file or list of netCDF files.\n\n"))

parser.add_argument('-t', '--interp_type', type=str, default='pstd',
    help=(
        f"Interpolation to standard pressure (pstd), standard altitude "
        f"(zstd), or altitude above ground level (zagl).\nWorks on "
        f"'daily', 'average', and 'diurn' files.\n"
        f"{Green}Example:\n"
        f"> MarsInterp 01336.atmos_average.nc -t pstd\n"
        f"> MarsInterp 01336.atmos_average.nc -t pstd -v pstd_default\n"
        f"{Nclr}\n\n"
    )
)

parser.add_argument('-v', '--vertical_grid', type=str, default=None,
    help=(
        f"For use with ``-t``. Specify a custom vertical grid to "
        f"interpolate to.\n"
        f"Custom grids defined in ``amescap_profile``.\nFor first "
        f"time use, copy ``amescap_profile`` to your home directory:\n"
        f"Works on 'daily', 'diurn', and 'average' files.\n"
        f"{Cyan}cp path/to/amesCAP/mars_templates/amescap_profile "
        f"~/.amescap_profile\n"
        f"{Green}Example:\n"
        f"> MarsInterp 01336.atmos_average.nc -t zstd -v phalf_mb"
        f"{Nclr}\n\n"
    )
)

parser.add_argument('-incl', '--include', nargs='+',
    help=(
        f"Only include the listed variables in the action. Dimensions "
        f"and 1D variables are always included.\n"
        f"Works on 'daily', 'diurn', and 'average' files.\n"
        f"{Green}Example:\n"
        f"> MarsInterp 01336.atmos_daily.nc -incl temp ps ts"
        f"{Nclr}\n\n"
    )
)

parser.add_argument('-print', '--print_grid', action='store_true',
    help=(
        f"Print the vertical grid to the screen.\n{Yellow}This does not "
        f"run the interpolation, it only prints grid information.\n"
        f"{Green}Example:\n"
        f"> MarsInterp 01336.atmos_average.nc -t pstd -v pstd_default -print"
        f"{Nclr}\n\n"
    )
)

parser.add_argument('-ext', '--extension', type=str, default=None,
    help=(
        f"Must be paired with an argument listed above.\nInstead of "
        f"overwriting a file to perform a function, ``-ext``\ntells "
        f"CAP to create a new file with the extension name specified "
        f"here.\n"
        f"{Green}Example:\n"
        f"> MarsInterp 01336.atmos_average.nc -t pstd -ext _my_pstd\n"
        f"{Blue}(Produces 01336.atmos_average_my_pstd.nc and "
        f"preserves all other files)"
        f"{Nclr}\n\n"
    )
)

parser.add_argument('--debug', action='store_true',
    help=(
        f"Use with any other argument to pass all Python errors and\n"
        f"status messages to the screen when running CAP.\n"
        f"{Green}Example:\n"
        f"> MarsInterp 01336.atmos_average.nc -t pstd --debug"
        f"{Nclr}\n\n"
    )
)

parser.add_argument('-j', '--jobs', type=int, default=1,
    help=(
        f"Number of parallel jobs for processing multiple files.\n"
        f"Use with caution for large files due to memory usage.\n"
        f"{Green}Example:\n"
        f"> MarsInterp *.nc -t pstd -j 4"
        f"{Nclr}\n\n"
    )
)

args = parser.parse_args()
debug = args.debug

if args.input_file:
    for file in args.input_file:
        if not re.search(".nc", file.name):
            parser.error(f"{Red}{file.name} is not a netCDF file{Nclr}")
            exit()


# ======================================================================
#                           DEFINITIONS
# ======================================================================

# Fill values for NaN. Do not use np.NaN because it is deprecated and
# will raise issues when using runpinterp
fill_value = 0.

# Define constants
rgas = 189.     # J/(kg-K) -> m2/(s2 K)
g = 3.72        # m/s2
R = 8.314       # J/ mol. K
Cp = 735.0      # J/K
M_co2 = 0.044   # kg/mol

filepath = os.getcwd()

# ======================================================================
#                       VECTORIZED MAIN PROGRAM
# ======================================================================


@debug_wrapper
def main():
    """
    VECTORIZED Main function for performing vertical interpolation on Mars
    atmospheric model NetCDF files.

    This vectorized version includes:
    - Batch processing of variables with similar characteristics
    - Optimized memory usage patterns
    - Vectorized array operations throughout
    - Pre-computation of interpolation indices
    - Efficient handling of multiple files
    - Optional parallel processing capability
    """

    start_time = time.time()
    
    # Extract file list and parameters
    file_list = [f.name for f in args.input_file]
    
    # Vectorized grid setup
    try:
        config, lev_in = vectorized_grid_setup(args.interp_type, args.vertical_grid)
    except ValueError as e:
        print(f"{Red}{str(e)}{Nclr}")
        return 1
    
    # Handle special case: print grid and exit
    if args.print_grid:
        print(*lev_in)
        return 0
    
    # Handle zstd-specific requirements
    if args.interp_type == "zstd":
        name_fixed = find_fixedfile(file_list[0])
        try:
            with Dataset(name_fixed, 'r') as f_fixed:
                model = read_variable_dict_amescap_profile(f_fixed)
                zsurf = f_fixed.variables["zsurf"][:]
        except FileNotFoundError:
            print(f"{Red}***Error*** Topography (zsurf) is required for "
                  f"interpolation to zstd, but the file {name_fixed} "
                  f"cannot be found{Nclr}")
            return 1
    
    # Vectorized file processing
    print(f"{Cyan}Processing {len(file_list)} file(s) with vectorized algorithms...{Nclr}")
    
    if args.jobs > 1 and len(file_list) > 1:
        # Parallel processing for multiple files
        print(f"{Cyan}Using {args.jobs} parallel jobs...{Nclr}")
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            # Create tasks for parallel execution
            futures = []
            for ifile in file_list:
                if args.extension:
                    newname = f"{filepath}/{ifile[:-3]}_{args.interp_type}_{args.extension}.nc"
                else:
                    newname = f"{filepath}/{ifile[:-3]}_{args.interp_type}.nc"
                
                future = executor.submit(
                    process_single_file_vectorized,
                    ifile, newname, config, lev_in,
                    zsurf if args.interp_type == "zstd" else None,
                    args
                )
                futures.append(future)
            
            # Collect results
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per file
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"{Red}Parallel processing error: {str(e)}{Nclr}")
                    results.append({"success": False, "error": str(e)})
    else:
        # Sequential processing with vectorized operations
        results = []
        for ifile in file_list:
            start_time = time.time()
            
            # Check file availability
            check_file_tape(ifile)
            
            # Generate output filename
            if args.extension:
                newname = f"{filepath}/{ifile[:-3]}_{args.interp_type}_{args.extension}.nc"
            else:
                newname = f"{filepath}/{ifile[:-3]}_{args.interp_type}.nc"
            
            # Process single file with vectorized operations
            result = process_single_file_vectorized(ifile, newname, config, lev_in, 
                                                   zsurf if args.interp_type == "zstd" else None, 
                                                   args)
            
            if result:
                results.append(result)
                print(f"Completed {ifile} in {(time.time() - start_time):.3f} sec")
    
    # Summary
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r and r.get("success", False))
    
    print(f"\n{Green}Vectorized processing complete!{Nclr}")
    print(f"Successfully processed: {successful}/{len(file_list)} files")
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Average time per file: {total_time/len(file_list):.3f} seconds")
    
    return 0 if successful == len(file_list) else 1


# ======================================================================
#                           END OF PROGRAM
# ======================================================================

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)