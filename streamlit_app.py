import streamlit as st
import pandas as pd
import numpy as np
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from io import BytesIO
import tempfile
import zipfile

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

st.set_page_config(page_title="Reduced-Order-Model-ROM--Degradation", layout="wide")

# =========================================================
# Data classes
# =========================================================
@dataclass
class BuildingConfig:
    building_name: str = "Validation Building"
    building_type: str = "Educational / University building"
    weather_source_label: str = "User-defined"
    conditioned_area_m2: float = 5000.0
    floors: int = 4
    n_spaces: int = 40
    floor_to_floor_m: float = 3.2
    wall_u_value: float = 0.60
    roof_u_value: float = 0.35
    window_u_value: float = 2.70
    shgc: float = 0.35
    glazing_ratio: float = 0.30
    infiltration_ach: float = 0.50
    occupancy_density_p_m2: float = 0.08
    lighting_w_m2: float = 10.0
    equipment_w_m2: float = 8.0
    sensible_heat_per_person_w: float = 75.0
    aspect_ratio: float = 1.5

@dataclass
class HVACConfig:
    hvac_system_type: str = "Chiller_AHU"
    airflow_m3_h_m2: float = 4.0
    cooling_design_w_m2: float = 100.0
    heating_design_w_m2: float = 55.0
    cooling_cop: float = 3.2
    heating_efficiency: float = 0.88
    fan_total_efficiency: float = 0.60
    fan_static_pressure_pa: float = 650.0
    pump_specific_w_m2: float = 1.3
    auxiliary_w_m2: float = 0.55
    cooling_setpoint_c: float = 24.0
    heating_setpoint_c: float = 20.0
    weekend_occupancy_factor: float = 0.18
    electricity_co2_kg_kwh: float = 0.55
    gas_co2_kg_kwh: float = 0.20

@dataclass
class DegradationConfig:
    degradation_model: str = "Physics-based fouling/clogging"
    cop_aging_rate: float = 0.0050
    rf_star: float = 0.000200
    fouling_growth_B: float = 0.015
    dust_accumulation_rate: float = 1.20
    clogging_coefficient: float = 6.00
    degradation_trigger: float = 0.55
    linear_slope_per_day: float = 0.000120
    exponential_rate_per_day: float = 0.000180

SEVERITY_MULT = {"Mild":0.70,"Moderate":1.00,"Severe":1.35,"High":1.60}
STRATEGY_CFG = {
    "S0":{"maint_factor":0.00,"flow_recovery":0.00,"setpoint_shift_c":0.00},
    "S1":{"maint_factor":0.18,"flow_recovery":0.10,"setpoint_shift_c":0.20},
    "S2":{"maint_factor":0.34,"flow_recovery":0.18,"setpoint_shift_c":0.30},
    "S3":{"maint_factor":0.52,"flow_recovery":0.28,"setpoint_shift_c":0.40},
}
CLIMATE_LEVELS = {
    "C0_Baseline":{"temp_add":0.0,"solar_mult":1.00,"rh_add":0.0},
    "C1_Warm":{"temp_add":1.5,"solar_mult":1.03,"rh_add":1.0},
    "C2_Hot":{"temp_add":3.0,"solar_mult":1.06,"rh_add":2.0},
    "C3_Extreme":{"temp_add":5.0,"solar_mult":1.10,"rh_add":3.5},
}
HVAC_PRESETS = {
    "Chiller_AHU":{"cooling_cop":3.2,"heating_efficiency":0.88,"fan_total_efficiency":0.60,"fan_static_pressure_pa":650.0,"pump_specific_w_m2":1.3,"auxiliary_w_m2":0.55},
    "DX_Rooftop":{"cooling_cop":2.9,"heating_efficiency":0.84,"fan_total_efficiency":0.58,"fan_static_pressure_pa":700.0,"pump_specific_w_m2":0.4,"auxiliary_w_m2":0.35},
    "FCU_Boiler_Chiller":{"cooling_cop":3.4,"heating_efficiency":0.90,"fan_total_efficiency":0.62,"fan_static_pressure_pa":500.0,"pump_specific_w_m2":1.8,"auxiliary_w_m2":0.50},
    "VAV_Reheat":{"cooling_cop":3.3,"heating_efficiency":0.87,"fan_total_efficiency":0.61,"fan_static_pressure_pa":750.0,"pump_specific_w_m2":1.1,"auxiliary_w_m2":0.55},
    "Custom":None,
}
VALIDATION_METRICS = {
    "Energy": ["Total HVAC Energy (kWh)", "Energy Consumption (kWh)", "Cooling (Electricity)", "Total HVAC Energy", "energy_kwh_day", "Total Energy MWh"],
    "Comfort": ["Comfort Deviation", "Comfort Deviation Mean (C)", "Mean Comfort Deviation (C)", "comfort_dev_C", "Operative Temperature"],
    "Degradation": ["Mean Degradation Index", "Degradation Index"],
    "Carbon": ["Total CO2 Production (kg)", "Carbon Footprint (kgCO2)", "CO2 Production (kg)"],
    "Health": ["Building Health Index"],
}

BENCHMARK_PARAMETERS = {
    "wall_u_value": ("Wall U-value", 0.20),
    "roof_u_value": ("Roof U-value", 0.20),
    "window_u_value": ("Window U-value", 0.20),
    "glazing_ratio": ("Glazing ratio", 0.20),
    "infiltration_ach": ("Infiltration", 0.25),
    "occupancy_density_p_m2": ("Occupancy density", 0.20),
    "lighting_w_m2": ("Lighting power density", 0.20),
    "equipment_w_m2": ("Equipment power density", 0.20),
    "airflow_m3_h_m2": ("Airflow intensity", 0.20),
    "cooling_design_w_m2": ("Cooling design intensity", 0.15),
    "heating_design_w_m2": ("Heating design intensity", 0.15),
    "cooling_cop": ("Cooling COP", 0.15),
    "cop_aging_rate": ("COP aging rate", 0.25),
    "fouling_growth_B": ("Fouling growth constant B", 0.25),
    "dust_accumulation_rate": ("Dust accumulation rate", 0.25),
    "clogging_coefficient": ("Clogging coefficient", 0.25),
}

DEFAULT_SWITCHES = {
    "use_envelope": True,
    "use_walls": True,
    "use_roof": True,
    "use_windows": True,
    "use_solar": True,
    "use_infiltration": True,
    "use_internal_gains": True,
    "use_people_gains": True,
    "use_lighting_gains": True,
    "use_equipment_gains": True,
    "use_hvac_fans": True,
    "use_hvac_pumps": True,
    "use_hvac_aux": True,
    "use_heating": True,
    "use_cooling": True,
    "use_degradation": True,
    "use_catboost": True,
    "use_benchmark": True,
    "use_zone_analysis": True,
}

# =========================================================
# Helpers
# =========================================================
def infer_col(cols, candidates):
    mapping = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in mapping:
            return mapping[cand.lower()]
    return None

def read_csv_fallback(file_obj):
    for enc in ["utf-8","latin1","cp1252","ISO-8859-1"]:
        try:
            file_obj.seek(0)
            return pd.read_csv(file_obj, encoding=enc)
        except Exception:
            pass
    raise ValueError("Could not read CSV file.")

def aggregate_weather_to_daily(df):
    out = df.copy()
    out["date_only"] = pd.to_datetime(out["Date/Time"]).dt.floor("D")
    daily = out.groupby("date_only", as_index=False).agg(
        T_amb_C=("T_amb_C","mean"),
        RH_pct=("RH_pct","mean"),
        GHI_Wm2=("GHI_Wm2","sum"),
    )
    return daily.rename(columns={"date_only":"Date/Time"})

def read_epw(uploaded_file):
    content = uploaded_file.getvalue().decode("utf-8", errors="ignore").splitlines()
    rows = []
    for line in content[8:]:
        parts = line.split(",")
        if len(parts) < 14:
            continue
        try:
            year = int(float(parts[0])); month = int(float(parts[1])); day = int(float(parts[2])); hour = int(float(parts[3]))
            dry = float(parts[6]); rh = float(parts[8]); ghi = float(parts[13])
        except Exception:
            continue
        ts = pd.Timestamp(year=year, month=month, day=day, hour=max(min(hour-1,23),0))
        rows.append({"Date/Time":ts,"T_amb_C":dry,"RH_pct":rh,"GHI_Wm2":ghi})
    if not rows:
        raise ValueError("No valid rows parsed from EPW.")
    return aggregate_weather_to_daily(pd.DataFrame(rows))

def read_weather_csv(uploaded_file):
    df = read_csv_fallback(uploaded_file)
    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]].copy()
    d = infer_col(df.columns,["Date/Time","date","timestamp","datetime"])
    t = infer_col(df.columns,["Outdoor Dry-Bulb Temperature","Outside Dry-Bulb Temperature","T_amb_C","temperature","temp"])
    rh = infer_col(df.columns,["Relative Humidity","Outside Relative Humidity","RH_pct","humidity","rh"])
    ghi = infer_col(df.columns,["Global Solar Radiation","Global Horizontal Solar","GHI_Wm2","ghi"])
    direct = infer_col(df.columns,["Direct Normal Solar","Direct Solar","Direct Normal Radiation"])
    diffuse = infer_col(df.columns,["Diffuse Horizontal Solar","Diffuse Solar","Diffuse Radiation"])
    if d is None or t is None:
        raise ValueError("Weather CSV must contain date/time and dry-bulb temperature.")
    if ghi is not None:
        solar = pd.to_numeric(df[ghi], errors="coerce")
    else:
        solar = pd.Series([0.0]*len(df))
        if direct is not None:
            solar = solar + pd.to_numeric(df[direct], errors="coerce").fillna(0)
        if diffuse is not None:
            solar = solar + pd.to_numeric(df[diffuse], errors="coerce").fillna(0)
    rh_series = pd.to_numeric(df[rh], errors="coerce") if rh is not None else pd.Series([60.0]*len(df))
    out = pd.DataFrame({"Date/Time":pd.to_datetime(df[d], errors="coerce"),"T_amb_C":pd.to_numeric(df[t], errors="coerce"),"RH_pct":rh_series,"GHI_Wm2":solar}).dropna(subset=["Date/Time","T_amb_C"]).reset_index(drop=True)
    return aggregate_weather_to_daily(out)

def read_weather_auto(uploaded_file):
    return read_epw(uploaded_file) if uploaded_file.name.lower().endswith(".epw") else read_weather_csv(uploaded_file)

def geometry(b):
    floor_area = b.conditioned_area_m2/max(b.floors,1)
    depth = (floor_area/max(b.aspect_ratio,1e-9))**0.5
    width = floor_area/max(depth,1e-9)
    perimeter = 2*(width+depth)
    wall_area = perimeter*b.floor_to_floor_m*b.floors
    roof_area = floor_area
    window_area = wall_area*b.glazing_ratio
    opaque_wall_area = max(wall_area-window_area,0.0)
    volume = b.conditioned_area_m2*b.floor_to_floor_m
    return {"roof_area_m2":roof_area,"window_area_m2":window_area,"opaque_wall_area_m2":opaque_wall_area,"volume_m3":volume}

def weekend_factor(ts,h):
    return h.weekend_occupancy_factor if ts.weekday()>=5 else 1.0

def internal_gains_kw(b,occ,switches):
    q_people = b.conditioned_area_m2*b.occupancy_density_p_m2*b.sensible_heat_per_person_w*occ/1000.0 if switches["use_people_gains"] else 0.0
    q_lighting = b.conditioned_area_m2*b.lighting_w_m2*occ/1000.0 if switches["use_lighting_gains"] else 0.0
    q_equipment = b.conditioned_area_m2*b.equipment_w_m2*occ/1000.0 if switches["use_equipment_gains"] else 0.0
    total = (q_people+q_lighting+q_equipment) if switches["use_internal_gains"] else 0.0
    return {"q_people_kw":q_people,"q_lighting_kw":q_lighting,"q_equipment_kw":q_equipment,"q_internal_kw":total}

def solar_gains_kw(b,geom,ghi,switches):
    return ghi*geom["window_area_m2"]*b.shgc/1000.0 if switches["use_solar"] else 0.0

def compute_degradation_index(day_idx,t_amb,rh,severity_mult,maint_factor,dcfg,switches):
    if not switches["use_degradation"]:
        return 0.0
    if dcfg.degradation_model=="Linear time-series":
        base = dcfg.linear_slope_per_day*(day_idx+1)
    elif dcfg.degradation_model=="Exponential time-series":
        base = 1.0-math.exp(-dcfg.exponential_rate_per_day*(day_idx+1))
    else:
        time_term = 1.0-math.exp(-dcfg.fouling_growth_B*0.01*(day_idx+1))
        stress_term = 1.0+0.025*max(t_amb-28.0,0.0)+0.010*max(rh-60.0,0.0)
        fouling_term = 1.0+min(dcfg.rf_star*5000.0,2.0)
        dust_term = 1.0+0.05*dcfg.dust_accumulation_rate+0.01*dcfg.clogging_coefficient
        base = time_term*stress_term*fouling_term*dust_term*dcfg.cop_aging_rate*25.0
    return max(0.0,min(1.0,base*severity_mult*(1.0-0.75*maint_factor)))

def calc_cooling_cop(h,t_amb,deg,dcfg):
    base = max(1.5,h.cooling_cop*(1.0-0.015*max(t_amb-24.0,0.0)))
    cop = base*(1.0-dcfg.cop_aging_rate*10.0*deg) if dcfg.degradation_model!="Physics-based fouling/clogging" else base*(1.0-0.28*deg)
    return max(1.2,cop)

def calc_heating_eff(h,t_amb,deg):
    base = max(0.60,h.heating_efficiency*(1.0-0.010*max(20.0-t_amb,0.0)))
    return max(0.50,base*(1.0-0.16*deg))

def fan_energy_kwh_day(b,h,occ,deg,flow_recovery,dcfg,switches):
    if not switches["use_hvac_fans"]:
        return 0.0
    mech_air = h.airflow_m3_h_m2*b.conditioned_area_m2*max(occ,0.35)
    eff_press = h.fan_static_pressure_pa*(1.0+0.75*deg)*(1.0-0.25*flow_recovery)
    if dcfg.degradation_model=="Physics-based fouling/clogging" and switches["use_degradation"]:
        eff_press += min(240.0, dcfg.clogging_coefficient*dcfg.dust_accumulation_rate*3.0*(1.0+deg))
    fan_kw = (mech_air/3600.0)*eff_press/max(h.fan_total_efficiency*1000.0,1e-9)
    return fan_kw*24.0

def pump_energy_kwh_day(b,h,occ,deg,switches):
    if not switches["use_hvac_pumps"]:
        return 0.0
    return h.pump_specific_w_m2*b.conditioned_area_m2*max(occ,0.35)*(1.0+0.30*deg)*24.0/1000.0

def aux_energy_kwh_day(b,h,occ,switches):
    if not switches["use_hvac_aux"]:
        return 0.0
    return h.auxiliary_w_m2*b.conditioned_area_m2*max(occ,0.35)*24.0/1000.0

def apply_climate(weather, climate_key):
    c = CLIMATE_LEVELS[climate_key]
    out = weather.copy()
    out["T_amb_C"] = out["T_amb_C"]+c["temp_add"]
    out["RH_pct"] = out["RH_pct"]+c["rh_add"]
    out["GHI_Wm2"] = out["GHI_Wm2"]*c["solar_mult"]
    return out

def build_zone_table(df, n_zones, zone_names_text):
    if n_zones <= 0:
        return pd.DataFrame()
    names = [s.strip() for s in zone_names_text.split(",") if s.strip()]
    if len(names) < n_zones:
        names += [f"Zone_{i}" for i in range(len(names)+1, n_zones+1)]
    names = names[:n_zones]
    weights = np.linspace(0.85,1.15,n_zones)
    weights = weights/weights.sum()
    frames = []
    for idx,(name,w) in enumerate(zip(names,weights), start=1):
        z = df.copy()
        z["Zone"] = name
        z["Zone Energy (kWh)"] = z["Total HVAC Energy (kWh)"]*n_zones*w
        z["Zone Comfort Deviation"] = z["Comfort Deviation"]*(0.95+0.1*(idx-1)/max(n_zones-1,1))
        z["Zone Degradation Index"] = np.clip(z["Degradation Index"]*(0.96+0.08*(idx-1)/max(n_zones-1,1)),0,1)
        z["Zone Carbon Footprint (kgCO2)"] = z["CO2 Production (kg)"]*n_zones*w
        z["Zone Health Index"] = np.clip(z["Building Health Index"]*(1.02-0.04*(idx-1)/max(n_zones-1,1)),0,100)
        frames.append(z[["Date/Time","Scenario Key","Zone","Zone Energy (kWh)","Zone Comfort Deviation","Zone Degradation Index","Zone Carbon Footprint (kgCO2)","Zone Health Index"]])
    return pd.concat(frames, ignore_index=True)

def simulate(weather,b,h,dcfg,switches,severity=None,strategy=None,climate_key="C0_Baseline",baseline=False):
    geom=geometry(b); rows=[]; weather=apply_climate(weather,climate_key)
    wall_u = b.wall_u_value if switches["use_walls"] else 0.0
    roof_u = b.roof_u_value if switches["use_roof"] else 0.0
    win_u = b.window_u_value if switches["use_windows"] else 0.0
    ua_total_w_k = (wall_u*geom["opaque_wall_area_m2"]+roof_u*geom["roof_area_m2"]+win_u*geom["window_area_m2"]) if switches["use_envelope"] else 0.0
    rho_cp_kwh_m3k = 1.2*1.005/3600.0
    infil_air_m3_h = b.infiltration_ach*geom["volume_m3"] if switches["use_infiltration"] else 0.0
    severity_mult = 1.0 if baseline else SEVERITY_MULT[severity]
    strat = {"maint_factor":0.0,"flow_recovery":0.0,"setpoint_shift_c":0.0} if baseline else STRATEGY_CFG[strategy]
    for i,r in weather.reset_index(drop=True).iterrows():
        ts=r["Date/Time"]; t=float(r["T_amb_C"]); rh=float(r["RH_pct"]); ghi=float(r["GHI_Wm2"])
        occ=weekend_factor(ts,h)
        gains=internal_gains_kw(b,occ,switches)
        q_solar=solar_gains_kw(b,geom,ghi,switches)
        deg=0.0 if baseline else compute_degradation_index(i,t,rh,severity_mult,strat["maint_factor"],dcfg,switches)
        cool_sp=h.cooling_setpoint_c+strat["setpoint_shift_c"]; heat_sp=h.heating_setpoint_c-0.5*strat["setpoint_shift_c"]
        dt_cool=max(t-cool_sp,0.0); dt_heat=max(heat_sp-t,0.0)
        q_env_cool_kw=ua_total_w_k*dt_cool/1000.0; q_env_heat_kw=ua_total_w_k*dt_heat/1000.0
        mech_air=h.airflow_m3_h_m2*b.conditioned_area_m2*(1.0-0.18*deg+0.12*strat["flow_recovery"])
        q_vent_cool_kw=(mech_air+infil_air_m3_h)*rho_cp_kwh_m3k*dt_cool
        q_vent_heat_kw=(mech_air+infil_air_m3_h)*rho_cp_kwh_m3k*dt_heat
        q_cool_kw=max((q_env_cool_kw+q_vent_cool_kw+gains["q_internal_kw"]+q_solar)*(1.0+0.10*deg),0.0) if switches["use_cooling"] else 0.0
        q_heat_kw=max((q_env_heat_kw+q_vent_heat_kw-0.75*gains["q_internal_kw"]-q_solar)*(1.0+0.08*deg),0.0) if switches["use_heating"] else 0.0
        q_cool_kw=min(q_cool_kw,b.conditioned_area_m2*h.cooling_design_w_m2/1000.0)
        q_heat_kw=min(q_heat_kw,b.conditioned_area_m2*h.heating_design_w_m2/1000.0)
        cop_c=calc_cooling_cop(h,t,deg,dcfg)
        eff_h=calc_heating_eff(h,t,deg)
        cooling_kwh=q_cool_kw*24.0/max(cop_c,1e-9) if switches["use_cooling"] else 0.0
        heating_gas_kwh=q_heat_kw*24.0/max(eff_h,1e-9) if switches["use_heating"] else 0.0
        fan_kwh=fan_energy_kwh_day(b,h,occ,deg,strat["flow_recovery"],dcfg,switches)
        pump_kwh=pump_energy_kwh_day(b,h,occ,deg,switches)
        aux_kwh=aux_energy_kwh_day(b,h,occ,switches)
        total_hvac=cooling_kwh+heating_gas_kwh+fan_kwh+pump_kwh+aux_kwh
        operative_temp=22.5+0.18*(t-22.5)+0.0005*ghi+0.03*gains["q_internal_kw"]-0.05*q_cool_kw+0.04*q_heat_kw+0.75*deg-0.30*strat["flow_recovery"]
        comfort_dev=abs(operative_temp-24.0)
        co2_kg=((cooling_kwh+fan_kwh+pump_kwh+aux_kwh)*h.electricity_co2_kg_kwh+heating_gas_kwh*h.gas_co2_kg_kwh)
        building_health=max(0.0,100.0*(1.0-0.65*deg-0.10*min(comfort_dev/5.0,1.0)))
        rows.append({"Date/Time":ts,"Scenario Key":"BASELINE" if baseline else f"{strategy}_{severity}_{climate_key}","Strategy":"BASELINE" if baseline else strategy,"Severity":"None" if baseline else severity,"Climate":climate_key,"Degradation Model":dcfg.degradation_model,"Outdoor Dry-Bulb Temperature":t,"Relative Humidity":rh,"Global Solar Radiation":ghi,"Occupancy Factor":occ,"People Gains (kW)":gains["q_people_kw"],"Lighting Gains (kW)":gains["q_lighting_kw"],"Equipment Gains (kW)":gains["q_equipment_kw"],"Internal Gains Total (kW)":gains["q_internal_kw"],"Solar Gains (kW)":q_solar,"Transmission UA Load Proxy (kW)":q_env_cool_kw if q_env_cool_kw>0 else q_env_heat_kw,"Ventilation/Infiltration Load (kW)":q_vent_cool_kw if q_vent_cool_kw>0 else q_vent_heat_kw,"Cooling Load (kW)":q_cool_kw,"Heating Load (kW)":q_heat_kw,"Cooling Electricity (kWh)":cooling_kwh,"Heating (Gas) (kWh)":heating_gas_kwh,"System Fans (kWh)":fan_kwh,"System Pumps (kWh)":pump_kwh,"Auxiliary Energy (kWh)":aux_kwh,"Total HVAC Energy (kWh)":total_hvac,"Cooling COP":cop_c,"Heating Efficiency":eff_h,"Operative Temperature":operative_temp,"Comfort Deviation":comfort_dev,"Degradation Index":deg,"CO2 Production (kg)":co2_kg,"Building Health Index":building_health})
    return pd.DataFrame(rows)

def build_kpi_table(df):
    return pd.DataFrame([{
        "Energy Consumption (kWh)": float(df["Total HVAC Energy (kWh)"].sum()),
        "Comfort Deviation Mean (C)": float(df["Comfort Deviation"].mean()),
        "Mean Degradation Index": float(df["Degradation Index"].mean()),
        "Carbon Footprint (kgCO2)": float(df["CO2 Production (kg)"].sum()),
        "Building Health Index": float(df["Building Health Index"].mean()),
    }])

def run_catboost_analysis(df, forecast_days, enabled=True):
    if not enabled:
        return pd.DataFrame({"Model":["Disabled by user"]})
    features=df.copy()
    features["day_idx"]=np.arange(len(features))
    features["month"]=pd.to_datetime(features["Date/Time"]).dt.month
    X=features[["day_idx","month","Outdoor Dry-Bulb Temperature","Relative Humidity","Global Solar Radiation","Degradation Index","Occupancy Factor"]]
    horizon=min(forecast_days, len(features))
    future=features.tail(horizon).copy()
    future["day_idx"]=np.arange(len(features), len(features)+len(future))
    def _fallback(label):
        return pd.DataFrame({
            "Date/Time": pd.date_range(pd.to_datetime(features["Date/Time"].iloc[-1])+pd.Timedelta(days=1), periods=len(future), freq="D"),
            "Predicted Energy Consumption (kWh)": future["Total HVAC Energy (kWh)"].rolling(7, min_periods=1).mean().values,
            "Predicted Comfort Deviation (C)": future["Comfort Deviation"].rolling(7, min_periods=1).mean().values,
            "Predicted Mean Degradation Index": future["Degradation Index"].rolling(7, min_periods=1).mean().clip(0,1).values,
            "Predicted Carbon Footprint (kgCO2)": future["CO2 Production (kg)"].rolling(7, min_periods=1).mean().values,
            "Model": label
        })
    if not (CATBOOST_AVAILABLE and len(features) > 20):
        return _fallback("Fallback rolling mean")
    targets={"energy":features["Total HVAC Energy (kWh)"],"comfort":features["Comfort Deviation"],"deg":features["Degradation Index"],"co2":features["CO2 Production (kg)"]}
    for y in targets.values():
        if pd.Series(y).nunique(dropna=True) <= 1:
            return _fallback("Fallback constant-target")
    try:
        models={k:CatBoostRegressor(verbose=False, random_seed=42) for k in targets}
        for k in models:
            models[k].fit(X,targets[k])
        return pd.DataFrame({
            "Date/Time": pd.date_range(pd.to_datetime(features["Date/Time"].iloc[-1])+pd.Timedelta(days=1), periods=len(future), freq="D"),
            "Predicted Energy Consumption (kWh)": models["energy"].predict(future[X.columns]),
            "Predicted Comfort Deviation (C)": models["comfort"].predict(future[X.columns]),
            "Predicted Mean Degradation Index": np.clip(models["deg"].predict(future[X.columns]),0,1),
            "Predicted Carbon Footprint (kgCO2)": models["co2"].predict(future[X.columns]),
            "Model":"CatBoost"
        })
    except Exception:
        return _fallback("Fallback CatBoost error")

def benchmark_sensitivity(weather, b, h, dcfg, switches, climate_key, severity, strategy, enabled=True):
    if not enabled:
        return pd.DataFrame({"Status":["Disabled by user"]}), pd.DataFrame({"Status":["Disabled by user"]})
    base_df=simulate(weather,b,h,dcfg,switches,severity,strategy,climate_key,baseline=False)
    base_kpi=build_kpi_table(base_df).iloc[0]
    rows=[]
    for pname, (label, frac) in BENCHMARK_PARAMETERS.items():
        for direction, mult in [("Low", 1.0-frac), ("High", 1.0+frac)]:
            b2=BuildingConfig(**asdict(b)); h2=HVACConfig(**asdict(h)); d2=DegradationConfig(**asdict(dcfg))
            target=b2 if hasattr(b2,pname) else h2 if hasattr(h2,pname) else d2
            old=getattr(target,pname); new=max(1e-9, old*mult) if isinstance(old,(int,float)) else old
            setattr(target,pname,new)
            df=simulate(weather,b2,h2,d2,switches,severity,strategy,climate_key,baseline=False)
            kpi=build_kpi_table(df).iloc[0]
            rows.append({
                "Parameter": label, "Parameter Key": pname, "Case": direction, "Base Value": old, "Test Value": new,
                "Energy Consumption (kWh)": kpi["Energy Consumption (kWh)"],
                "Comfort Deviation Mean (C)": kpi["Comfort Deviation Mean (C)"],
                "Mean Degradation Index": kpi["Mean Degradation Index"],
                "Carbon Footprint (kgCO2)": kpi["Carbon Footprint (kgCO2)"],
                "Building Health Index": kpi["Building Health Index"],
                "Delta Energy %": 100.0*(kpi["Energy Consumption (kWh)"]-base_kpi["Energy Consumption (kWh)"])/max(base_kpi["Energy Consumption (kWh)"],1e-9),
                "Delta Comfort %": 100.0*(kpi["Comfort Deviation Mean (C)"]-base_kpi["Comfort Deviation Mean (C)"])/max(abs(base_kpi["Comfort Deviation Mean (C)"]),1e-9),
                "Delta Deg %": 100.0*(kpi["Mean Degradation Index"]-base_kpi["Mean Degradation Index"])/max(abs(base_kpi["Mean Degradation Index"]),1e-9),
                "Delta Carbon %": 100.0*(kpi["Carbon Footprint (kgCO2)"]-base_kpi["Carbon Footprint (kgCO2)"])/max(abs(base_kpi["Carbon Footprint (kgCO2)"]),1e-9),
                "Delta Health %": 100.0*(kpi["Building Health Index"]-base_kpi["Building Health Index"])/max(abs(base_kpi["Building Health Index"]),1e-9),
            })
    detail=pd.DataFrame(rows)
    summary=detail.groupby(["Parameter","Parameter Key"], as_index=False)[["Delta Energy %","Delta Comfort %","Delta Deg %","Delta Carbon %","Delta Health %"]].apply(lambda g: g.abs().mean()).reset_index(drop=True)
    summary["Overall Sensitivity Score"]=summary[["Delta Energy %","Delta Comfort %","Delta Deg %","Delta Carbon %","Delta Health %"]].abs().mean(axis=1)
    return detail, summary.sort_values("Overall Sensitivity Score", ascending=False).reset_index(drop=True)

def load_validation_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return {"sheet_1": read_csv_fallback(uploaded_file)}
    if name.endswith(".xlsx") or name.endswith(".xls"):
        uploaded_file.seek(0)
        xls = pd.ExcelFile(uploaded_file)
        result = {}
        for sheet in xls.sheet_names:
            uploaded_file.seek(0)
            result[sheet] = pd.read_excel(uploaded_file, sheet_name=sheet)
        return result
    raise ValueError("Validation upload must be CSV or Excel.")

def find_metric_column(df, metric_name):
    for cand in VALIDATION_METRICS[metric_name]:
        c = infer_col(df.columns, [cand])
        if c is not None:
            return c
    return None

def summarize_external_sources(sheet_map, source_label):
    rows = []
    for sheet_name, df in sheet_map.items():
        clean = df[[c for c in df.columns if not str(c).startswith("Unnamed")]].copy()
        row = {"Source": source_label, "Sheet": sheet_name}
        for metric in VALIDATION_METRICS.keys():
            col = find_metric_column(clean, metric)
            if col is None:
                row[metric] = np.nan
            else:
                vals = pd.to_numeric(clean[col], errors="coerce").dropna()
                row[metric] = float(vals.mean()) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def build_validation_table(app_summary, db_summary=None, pub_summary=None):
    app_row = {
        "Source": "ROM App",
        "Energy": float(app_summary["Total HVAC Energy (kWh)"].iloc[0]),
        "Comfort": float(app_summary["Mean Comfort Deviation (C)"].iloc[0]),
        "Degradation": float(app_summary["Mean Degradation Index"].iloc[0]),
        "Carbon": float(app_summary["Total CO2 Production (kg)"].iloc[0]),
        "Health": float(app_summary["Building Health Index"].iloc[0]),
    }
    rows=[app_row]
    if db_summary is not None and not db_summary.empty:
        for _,r in db_summary.iterrows():
            rows.append({"Source": f"DesignBuilder - {r['Sheet']}", "Energy": r["Energy"], "Comfort": r["Comfort"], "Degradation": r["Degradation"], "Carbon": r["Carbon"], "Health": r["Health"]})
    if pub_summary is not None and not pub_summary.empty:
        for _,r in pub_summary.iterrows():
            rows.append({"Source": f"Published - {r['Sheet']}", "Energy": r["Energy"], "Comfort": r["Comfort"], "Degradation": r["Degradation"], "Carbon": r["Carbon"], "Health": r["Health"]})
    out=pd.DataFrame(rows)
    base=out.iloc[0]
    for metric in ["Energy","Comfort","Degradation","Carbon","Health"]:
        out[f"{metric} Deviation % vs App"] = 100.0*(out[metric]-base[metric])/(abs(base[metric]) if abs(base[metric]) > 1e-9 else 1.0)
    return out

def build_outputs(all_data, zone_data=None, catboost_df=None, kpi_df=None, benchmark_detail=None, benchmark_summary=None, validation_df=None):
    fuel_breakdown=all_data[["Date/Time","System Fans (kWh)","System Pumps (kWh)","Auxiliary Energy (kWh)","Heating (Gas) (kWh)","Cooling Electricity (kWh)","Total HVAC Energy (kWh)"]].rename(columns={"System Fans (kWh)":"System Fans","System Pumps (kWh)":"System Pumps","Auxiliary Energy (kWh)":"Auxiliary Energy","Heating (Gas) (kWh)":"Heating (Gas)","Cooling Electricity (kWh)":"Cooling (Electricity)","Total HVAC Energy (kWh)":"Total HVAC Energy"})
    fuels_total=pd.DataFrame([{"Cooling (Electricity)":float(fuel_breakdown["Cooling (Electricity)"].sum()),"Heating (Gas)":float(fuel_breakdown["Heating (Gas)"].sum()),"System Fans":float(fuel_breakdown["System Fans"].sum()),"System Pumps":float(fuel_breakdown["System Pumps"].sum()),"Auxiliary Energy":float(fuel_breakdown["Auxiliary Energy"].sum()),"Total HVAC Energy":float(fuel_breakdown["Total HVAC Energy"].sum())}])
    comfort=all_data[["Date/Time","Operative Temperature","Comfort Deviation"]].copy()
    system_loads=all_data[["Date/Time","Cooling Load (kW)","Heating Load (kW)","Cooling COP","Heating Efficiency"]].copy()
    internal_gains=all_data[["Date/Time","People Gains (kW)","Lighting Gains (kW)","Equipment Gains (kW)","Internal Gains Total (kW)","Solar Gains (kW)"]].copy()
    site_data=all_data[["Date/Time","Outdoor Dry-Bulb Temperature","Relative Humidity","Global Solar Radiation","Occupancy Factor"]].copy()
    fabrics_vent=all_data[["Date/Time","Transmission UA Load Proxy (kW)","Ventilation/Infiltration Load (kW)"]].copy()
    co2_prod=all_data[["Date/Time","CO2 Production (kg)"]].copy()
    summary=pd.DataFrame([{"Cooling Electricity Total (kWh)":float(all_data["Cooling Electricity (kWh)"].sum()),"Heating Gas Total (kWh)":float(all_data["Heating (Gas) (kWh)"].sum()),"Fans Total (kWh)":float(all_data["System Fans (kWh)"].sum()),"Pumps Total (kWh)":float(all_data["System Pumps (kWh)"].sum()),"Aux Total (kWh)":float(all_data["Auxiliary Energy (kWh)"].sum()),"Total HVAC Energy (kWh)":float(all_data["Total HVAC Energy (kWh)"].sum()),"Mean Cooling Load (kW)":float(all_data["Cooling Load (kW)"].mean()),"Mean Heating Load (kW)":float(all_data["Heating Load (kW)"].mean()),"Mean Operative Temperature (C)":float(all_data["Operative Temperature"].mean()),"Mean Comfort Deviation (C)":float(all_data["Comfort Deviation"].mean()),"Mean Degradation Index":float(all_data["Degradation Index"].mean()),"Mean Cooling COP":float(all_data["Cooling COP"].mean()),"Total CO2 Production (kg)":float(all_data["CO2 Production (kg)"].sum()),"Building Health Index":float(all_data["Building Health Index"].mean())}])
    outputs={"ALL DATA.csv":all_data,"FUEL BREAKDOWN.csv":fuel_breakdown,"FUELS TOTAL.csv":fuels_total,"COMFORT.csv":comfort,"SYSTEM LOADS.csv":system_loads,"INTERNAL GAINS.csv":internal_gains,"SITE DATA.csv":site_data,"FABRICS AND VENTILATIONS.csv":fabrics_vent,"CO2 PRODUCTION.csv":co2_prod,"SUMMARY.csv":summary}
    if zone_data is not None and not zone_data.empty: outputs["ZONES ANALYSIS.csv"]=zone_data
    if catboost_df is not None and not catboost_df.empty: outputs["CATBOOST ANALYSIS.csv"]=catboost_df
    if kpi_df is not None and not kpi_df.empty: outputs["KPIS.csv"]=kpi_df
    if benchmark_detail is not None and not benchmark_detail.empty: outputs["EARLY BENCHMARK DETAIL.csv"]=benchmark_detail
    if benchmark_summary is not None and not benchmark_summary.empty: outputs["EARLY BENCHMARK SUMMARY.csv"]=benchmark_summary
    if validation_df is not None and not validation_df.empty: outputs["VALIDATION COMPARISON.csv"]=validation_df
    return outputs

def to_excel_bytes(all_outputs):
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path=Path(tmp.name)
    with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
        for sheet_name, df in all_outputs.items():
            df.to_excel(writer, sheet_name=sheet_name.replace(".csv","")[:31], index=False)
    data=tmp_path.read_bytes(); tmp_path.unlink(missing_ok=True); return data

def to_zip_bytes(all_outputs, config_obj):
    buf=BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as zf:
        for name, df in all_outputs.items():
            zf.writestr(name, df.to_csv(index=False))
        zf.writestr("run_config.json", json.dumps(config_obj, indent=2))
    return buf.getvalue()

# =========================================================
# UI
# =========================================================
st.title("Reduced-Order-Model-ROM--Degradation")
st.caption("More user-friendly version with feature switches and a dedicated zones CSV table.")

weather_file = st.file_uploader("Weather file (CSV or EPW)", type=["csv","epw"])
designbuilder_validation_file = st.file_uploader("Optional DesignBuilder validation upload (CSV/XLSX)", type=["csv","xlsx","xls"])
published_validation_file = st.file_uploader("Optional published dataset validation upload (CSV/XLSX)", type=["csv","xlsx","xls"])
axis_mode = st.selectbox("Analysis mode", ["Baseline only","One-axis Severity","One-axis Strategy","Two-axis Severity–Strategy","Three-axis Severity–Strategy–Climate"])

with st.expander("Quick controls", expanded=True):
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        use_zone_analysis = st.checkbox("Enable zone analysis", value=True)
        use_catboost = st.checkbox("Enable CatBoost analysis", value=True)
    with c2:
        use_benchmark = st.checkbox("Enable early benchmark", value=True)
        use_degradation = st.checkbox("Enable degradation", value=True)
    with c3:
        use_heating = st.checkbox("Include heating", value=True)
        use_cooling = st.checkbox("Include cooling", value=True)
    with c4:
        use_internal_gains = st.checkbox("Include internal gains", value=True)
        use_envelope = st.checkbox("Include envelope loads", value=True)

with st.expander("Building identity", expanded=True):
    c1,c2=st.columns(2)
    building_name=c1.text_input("Building name","Validation Building")
    building_type=c1.text_input("Building type","Educational / University building")
    weather_source_label=c2.text_input("Location / Weather source label","User-defined")

with st.expander("Geometry", expanded=True):
    c1,c2,c3,c4=st.columns(4)
    conditioned_area_m2=c1.number_input("Conditioned area (m²)", value=5000.0)
    floors=c2.number_input("Floors", min_value=1, value=4)
    n_spaces=c3.number_input("Number of spaces", min_value=1, value=40)
    zone_count=c4.number_input("Zones for analysis", min_value=1, value=4)
    zone_names_text=st.text_input("Zone names CSV", "North Zone, South Zone, East Zone, West Zone")

with st.expander("Parameter switches", expanded=False):
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        use_walls = st.checkbox("Use walls", value=True)
        use_roof = st.checkbox("Use roof", value=True)
        use_windows = st.checkbox("Use windows", value=True)
        use_solar = st.checkbox("Use solar gains", value=True)
    with col2:
        use_infiltration = st.checkbox("Use infiltration", value=True)
        use_people_gains = st.checkbox("Use people gains", value=True)
        use_lighting_gains = st.checkbox("Use lighting gains", value=True)
        use_equipment_gains = st.checkbox("Use equipment gains", value=True)
    with col3:
        use_hvac_fans = st.checkbox("Use fans", value=True)
        use_hvac_pumps = st.checkbox("Use pumps", value=True)
        use_hvac_aux = st.checkbox("Use auxiliary", value=True)
    with col4:
        st.caption("These switches let you include or exclude each parameter block from the calculation.")

with st.expander("Envelope", expanded=False):
    c1,c2,c3=st.columns(3)
    wall_u_value=c1.number_input("Wall U-value (W/m²K)", value=0.60)
    roof_u_value=c1.number_input("Roof U-value (W/m²K)", value=0.35)
    window_u_value=c2.number_input("Window U-value (W/m²K)", value=2.70)
    shgc=c2.number_input("SHGC", value=0.35)
    glazing_ratio=c3.number_input("Glazing ratio", value=0.30)
    infiltration_ach=c3.number_input("Infiltration (ACH)", value=0.50)

with st.expander("Internal loads", expanded=False):
    c1,c2=st.columns(2)
    occupancy_density_p_m2=c1.number_input("General occupancy density (person/m²)", value=0.08)
    lighting_w_m2=c1.number_input("Lighting power density (W/m²)", value=10.00)
    equipment_w_m2=c2.number_input("Equipment power density (W/m²)", value=8.00)
    sensible_heat_per_person_w=c2.number_input("Sensible heat per person (W)", value=75.00)

with st.expander("HVAC sizing and component", expanded=False):
    hvac_preset=st.selectbox("HVAC selector", list(HVAC_PRESETS.keys()), index=0)
    preset=HVAC_PRESETS[hvac_preset] or {}
    c1,c2,c3=st.columns(3)
    hvac_system_type=c1.text_input("HVAC system type", hvac_preset)
    airflow_m3_h_m2=c1.number_input("Airflow intensity (m³/h·m²)", value=4.00)
    cooling_design_w_m2=c2.number_input("Cooling design intensity (W/m²)", value=100.00)
    heating_design_w_m2=c2.number_input("Heating design intensity (W/m²)", value=55.00)
    cooling_cop_input=c3.number_input("Cooling COP", value=float(preset.get("cooling_cop",3.2)))
    heating_efficiency_input=c3.number_input("Heating efficiency", value=float(preset.get("heating_efficiency",0.88)))
    c4,c5,c6=st.columns(3)
    fan_total_efficiency_input=c4.number_input("Fan total efficiency", value=float(preset.get("fan_total_efficiency",0.60)))
    fan_static_pressure_pa_input=c4.number_input("Fan static pressure (Pa)", value=float(preset.get("fan_static_pressure_pa",650.0)))
    pump_specific_w_m2_input=c5.number_input("Pump specific (W/m²)", value=float(preset.get("pump_specific_w_m2",1.3)))
    auxiliary_w_m2_input=c5.number_input("Auxiliary (W/m²)", value=float(preset.get("auxiliary_w_m2",0.55)))
    cooling_setpoint_c_input=c6.number_input("Cooling setpoint (°C)", value=24.0)
    heating_setpoint_c_input=c6.number_input("Heating setpoint (°C)", value=20.0)

with st.expander("Degradation parameters", expanded=False):
    c1,c2,c3=st.columns(3)
    cop_aging_rate=c1.number_input("COP aging rate", value=0.0050, format="%.6f")
    rf_star=c1.number_input("RF* (fouling asymptote)", value=0.000200, format="%.6f")
    fouling_growth_B=c1.number_input("Fouling growth constant B", value=0.015, format="%.6f")
    dust_accumulation_rate=c2.number_input("Dust accumulation rate", value=1.20, format="%.4f")
    clogging_coefficient=c2.number_input("Clogging coefficient", value=6.00, format="%.4f")
    degradation_trigger=c2.number_input("Degradation trigger", value=0.55, format="%.4f")
    degradation_model=c3.selectbox("Degradation model",["Physics-based fouling/clogging","Linear time-series","Exponential time-series"],index=0)
    linear_slope_per_day=c3.number_input("Linear degradation slope per day", value=0.000120, format="%.6f")
    exponential_rate_per_day=c3.number_input("Exponential degradation rate per day", value=0.000180, format="%.6f")

with st.expander("Simulation controls", expanded=False):
    c1,c2,c3,c4=st.columns(4)
    severity_pick=c1.selectbox("Fixed severity", list(SEVERITY_MULT.keys()), index=1)
    strategy_pick=c2.selectbox("Fixed strategy", list(STRATEGY_CFG.keys()), index=0)
    climate_pick=c3.selectbox("Fixed climate", list(CLIMATE_LEVELS.keys()), index=0)
    simulation_years=c4.number_input("Simulation years", min_value=1, value=5)
    forecast_days=st.number_input("CatBoost forecast days", min_value=7, value=30)

run=st.button("Run user-friendly simulation", type="primary")

if weather_file is None:
    st.info("Upload a weather file to start.")
elif run:
    try:
        switches = {
            "use_envelope": use_envelope,
            "use_walls": use_walls,
            "use_roof": use_roof,
            "use_windows": use_windows,
            "use_solar": use_solar,
            "use_infiltration": use_infiltration,
            "use_internal_gains": use_internal_gains,
            "use_people_gains": use_people_gains,
            "use_lighting_gains": use_lighting_gains,
            "use_equipment_gains": use_equipment_gains,
            "use_hvac_fans": use_hvac_fans,
            "use_hvac_pumps": use_hvac_pumps,
            "use_hvac_aux": use_hvac_aux,
            "use_heating": use_heating,
            "use_cooling": use_cooling,
            "use_degradation": use_degradation,
            "use_catboost": use_catboost,
            "use_benchmark": use_benchmark,
            "use_zone_analysis": use_zone_analysis,
        }
        weather=read_weather_auto(weather_file)
        b=BuildingConfig(building_name=building_name,building_type=building_type,weather_source_label=weather_source_label,conditioned_area_m2=conditioned_area_m2,floors=int(floors),n_spaces=int(n_spaces),wall_u_value=wall_u_value,roof_u_value=roof_u_value,window_u_value=window_u_value,shgc=shgc,glazing_ratio=glazing_ratio,infiltration_ach=infiltration_ach,occupancy_density_p_m2=occupancy_density_p_m2,lighting_w_m2=lighting_w_m2,equipment_w_m2=equipment_w_m2,sensible_heat_per_person_w=sensible_heat_per_person_w)
        h=HVACConfig(hvac_system_type=hvac_system_type,airflow_m3_h_m2=airflow_m3_h_m2,cooling_design_w_m2=cooling_design_w_m2,heating_design_w_m2=heating_design_w_m2,cooling_cop=cooling_cop_input,heating_efficiency=heating_efficiency_input,fan_total_efficiency=fan_total_efficiency_input,fan_static_pressure_pa=fan_static_pressure_pa_input,pump_specific_w_m2=pump_specific_w_m2_input,auxiliary_w_m2=auxiliary_w_m2_input,cooling_setpoint_c=cooling_setpoint_c_input,heating_setpoint_c=heating_setpoint_c_input)
        dcfg=DegradationConfig(degradation_model=degradation_model,cop_aging_rate=cop_aging_rate,rf_star=rf_star,fouling_growth_B=fouling_growth_B,dust_accumulation_rate=dust_accumulation_rate,clogging_coefficient=clogging_coefficient,degradation_trigger=degradation_trigger,linear_slope_per_day=linear_slope_per_day,exponential_rate_per_day=exponential_rate_per_day)

        all_outputs={}; summary_frames=[]; scenario_data_map={}

        baseline_df=simulate(weather,b,h,dcfg,switches,None,None,"C0_Baseline",baseline=True)
        scenario_data_map["BASELINE"]=baseline_df
        baseline_zone=build_zone_table(baseline_df, int(zone_count), zone_names_text) if switches["use_zone_analysis"] else pd.DataFrame()
        baseline_kpi=build_kpi_table(baseline_df)
        baseline_cb=run_catboost_analysis(baseline_df, int(forecast_days), switches["use_catboost"])
        baseline_outputs=build_outputs(baseline_df, baseline_zone, baseline_cb, baseline_kpi)
        for k,v in baseline_outputs.items():
            all_outputs[f"BASELINE_{k}"]=v
        bs=baseline_outputs["SUMMARY.csv"].copy()
        bs.insert(0,"Scenario Key","BASELINE"); bs.insert(1,"Strategy","BASELINE"); bs.insert(2,"Severity","None"); bs.insert(3,"Climate","C0_Baseline"); bs.insert(4,"Degradation Model", dcfg.degradation_model)
        summary_frames.append(bs)

        if axis_mode=="One-axis Severity":
            scenarios=[(strategy_pick, sev, climate_pick) for sev in SEVERITY_MULT.keys()]
        elif axis_mode=="One-axis Strategy":
            scenarios=[(stg, severity_pick, climate_pick) for stg in STRATEGY_CFG.keys()]
        elif axis_mode=="Two-axis Severity–Strategy":
            scenarios=[(stg, sev, climate_pick) for stg in STRATEGY_CFG.keys() for sev in SEVERITY_MULT.keys()]
        elif axis_mode=="Three-axis Severity–Strategy–Climate":
            scenarios=[(stg, sev, clm) for stg in STRATEGY_CFG.keys() for sev in SEVERITY_MULT.keys() for clm in CLIMATE_LEVELS.keys()]
        else:
            scenarios=[]

        for strategy,severity,climate in scenarios:
            key=f"{strategy}_{severity}_{climate}"
            df=simulate(weather,b,h,dcfg,switches,severity,strategy,climate,baseline=False)
            scenario_data_map[key]=df
            zone_df=build_zone_table(df, int(zone_count), zone_names_text) if switches["use_zone_analysis"] else pd.DataFrame()
            kpi_df=build_kpi_table(df)
            cb_df=run_catboost_analysis(df, int(forecast_days), switches["use_catboost"])
            bench_detail=None; bench_summary=None
            if switches["use_benchmark"]:
                bench_detail, bench_summary = benchmark_sensitivity(weather,b,h,dcfg,switches,climate,severity,strategy,True)
            outputs=build_outputs(df, zone_df, cb_df, kpi_df, bench_detail, bench_summary)
            for k,v in outputs.items():
                all_outputs[f"{key}_{k}"]=v
            s=outputs["SUMMARY.csv"].copy()
            s.insert(0,"Scenario Key",key); s.insert(1,"Strategy",strategy); s.insert(2,"Severity",severity); s.insert(3,"Climate",climate); s.insert(4,"Degradation Model",dcfg.degradation_model)
            summary_frames.append(s)

        scenario_summary=pd.concat(summary_frames, ignore_index=True)
        all_outputs["scenario_summary_table.csv"]=scenario_summary

        db_summary=None; pub_summary=None; validation_df=None
        if designbuilder_validation_file is not None:
            db_summary=summarize_external_sources(load_validation_file(designbuilder_validation_file),"DesignBuilder")
        if published_validation_file is not None:
            pub_summary=summarize_external_sources(load_validation_file(published_validation_file),"Published")
        if db_summary is not None or pub_summary is not None:
            validation_df=build_validation_table(baseline_outputs["SUMMARY.csv"], db_summary, pub_summary)
            all_outputs["VALIDATION COMPARISON.csv"]=validation_df

        t1,t2,t3,t4,t5=st.columns(5)
        t1.metric("Weather days", len(weather))
        t2.metric("Scenarios", max(len(summary_frames)-1,0))
        t3.metric("Baseline HVAC (kWh)", round(float(baseline_outputs["SUMMARY.csv"]["Total HVAC Energy (kWh)"].iloc[0]),2))
        t4.metric("Building health", round(float(baseline_df["Building Health Index"].mean()),2))
        t5.metric("Zones table", "On" if switches["use_zone_analysis"] else "Off")

        sim_tab, zone_tab, bench_tab, cat_tab, val_tab, export_tab = st.tabs([
            "Simulation summary",
            "Zones table",
            "Early benchmark",
            "CatBoost analysis",
            "Validation upload",
            "Exports"
        ])

        with sim_tab:
            st.subheader("Framework summary table")
            st.dataframe(scenario_summary, use_container_width=True)
            st.line_chart(scenario_summary.set_index("Scenario Key")["Total HVAC Energy (kWh)"])
            st.line_chart(scenario_summary.set_index("Scenario Key")["Mean Comfort Deviation (C)"])
            st.line_chart(scenario_summary.set_index("Scenario Key")["Mean Degradation Index"])
            st.line_chart(scenario_summary.set_index("Scenario Key")["Total CO2 Production (kg)"])
            st.line_chart(scenario_summary.set_index("Scenario Key")["Building Health Index"])

        with zone_tab:
            if switches["use_zone_analysis"]:
                selected_scenario=st.selectbox("Select scenario for zones table", list(scenario_data_map.keys()), key="zones_scenario")
                zone_preview=build_zone_table(scenario_data_map[selected_scenario], int(zone_count), zone_names_text)
                st.dataframe(zone_preview, use_container_width=True)
                st.download_button("Download selected zones CSV", data=zone_preview.to_csv(index=False), file_name=f"{selected_scenario}_zones_table.csv", mime="text/csv")
                st.line_chart(zone_preview.pivot_table(index="Date/Time", columns="Zone", values="Zone Energy (kWh)", aggfunc="mean"))
                st.line_chart(zone_preview.pivot_table(index="Date/Time", columns="Zone", values="Zone Comfort Deviation", aggfunc="mean"))
            else:
                st.info("Zone analysis is disabled.")

        with bench_tab:
            if switches["use_benchmark"]:
                non_base_keys=[k for k in scenario_data_map.keys() if k != "BASELINE"]
                if non_base_keys:
                    selected_bench=st.selectbox("Select scenario for benchmark", non_base_keys, key="bench_scenario")
                    parts=selected_bench.split("_",2)
                    strategy=parts[0]; severity=parts[1]; climate="_".join(parts[2:])
                    bench_detail, bench_summary=benchmark_sensitivity(weather,b,h,dcfg,switches,climate,severity,strategy,True)
                    st.dataframe(bench_summary, use_container_width=True)
                    st.bar_chart(bench_summary.set_index("Parameter")["Overall Sensitivity Score"])
                    metric_view=st.selectbox("Sensitivity KPI curve", ["Delta Energy %","Delta Comfort %","Delta Deg %","Delta Carbon %","Delta Health %"])
                    st.line_chart(bench_detail.pivot_table(index="Parameter", columns="Case", values=metric_view, aggfunc="mean"))
                else:
                    st.info("Run a non-baseline scenario to see benchmark sensitivity.")
            else:
                st.info("Benchmark analysis is disabled.")

        with cat_tab:
            if switches["use_catboost"]:
                selected_cat=st.selectbox("Select scenario for CatBoost preview", list(scenario_data_map.keys()), key="cat_scenario")
                cb_preview=run_catboost_analysis(scenario_data_map[selected_cat], int(forecast_days), True)
                st.dataframe(cb_preview.head(150), use_container_width=True)
                if "Date/Time" in cb_preview.columns and "Predicted Energy Consumption (kWh)" in cb_preview.columns:
                    st.line_chart(cb_preview.set_index("Date/Time")[["Predicted Energy Consumption (kWh)","Predicted Comfort Deviation (C)","Predicted Mean Degradation Index","Predicted Carbon Footprint (kgCO2)"]])
            else:
                st.info("CatBoost analysis is disabled.")

        with val_tab:
            if validation_df is None:
                st.info("Upload DesignBuilder and/or published validation files to compare KPIs.")
            else:
                st.dataframe(validation_df, use_container_width=True)
                for metric in ["Energy","Comfort","Degradation","Carbon","Health"]:
                    st.line_chart(validation_df.set_index("Source")[metric])

        with export_tab:
            selected_sheet=st.selectbox("Select output sheet", list(all_outputs.keys()))
            st.dataframe(all_outputs[selected_sheet].head(250), use_container_width=True)
            config_export={"building":asdict(b),"hvac":asdict(h),"degradation":asdict(dcfg),"switches":switches,"axis_mode":axis_mode,"fixed_severity":severity_pick,"fixed_strategy":strategy_pick,"fixed_climate":climate_pick,"simulation_years":simulation_years,"forecast_days":int(forecast_days),"zones":int(zone_count),"zone_names":zone_names_text}
            excel_bytes=to_excel_bytes(all_outputs); zip_bytes=to_zip_bytes(all_outputs, config_export)
            d1,d2,d3=st.columns(3)
            d1.download_button("Download Excel workbook", data=excel_bytes, file_name="rom_degradation_user_friendly.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            d2.download_button("Download ZIP of all CSV outputs", data=zip_bytes, file_name="rom_degradation_user_friendly.zip", mime="application/zip")
            d3.download_button("Download config JSON", data=json.dumps(config_export, indent=2), file_name="rom_degradation_user_friendly.json", mime="application/json")
    except Exception as e:
        st.error(f"Simulation failed: {e}")
