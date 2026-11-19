# Parse NDB open data Q20 (sleep) secondary-area x sex x age tables for FY2018-2023
import pandas as pd, numpy as np, sys
R,O=sys.argv[1],sys.argv[2]
FILES={2018:('第6回','000824189'),2019:('第7回','001262550'),2020:('第8回','001259064'),2021:('第9回','001258904'),2022:('第10回','001495821'),2023:('第11回','r11_q20_area')}
ages=['40-44','45-49','50-54','55-59','60-64','65-69','70-74']
cols=[(s,a,4+(0 if s=='M' else 8)+i) for s in ('M','F') for i,a in enumerate(ages)]
out=[]
for fy,(rnd,f) in FILES.items():
    d=pd.read_excel(f'{R}/{f}.xlsx',header=None)
    assert str(fy) in str(d.iloc[0,0]) or (fy==2018 and 'H30' in str(d.iloc[0,0])), d.iloc[0,0]
    rows=[];pref=code=name=None
    for i in range(5,len(d)):
        r=d.iloc[i]
        if isinstance(r[0],str): pref=r[0]
        if pd.notna(r[1]): code=str(r[1]).zfill(4); name=r[2]
        if pref=='二次医療圏判別不可': code,name='9999','判別不可'
        for s,a,j in cols: rows.append((pref,code,name,r[3],s,a,r[j]))
    L=pd.DataFrame(rows,columns=['pref','area','area_nm','ans','sex','age','n'])
    L['n']=pd.to_numeric(L.n.replace({'‐':np.nan,'-':np.nan}),errors='coerce')
    W=L.pivot_table(index=['pref','area','area_nm','sex','age'],columns='ans',values='n',aggfunc='first').reset_index()
    W.columns.name=None; W=W.rename(columns={'はい':'yes','いいえ':'no'}); W.insert(0,'fy',fy); W.insert(1,'round',rnd)
    W=W.dropna(subset=['area'])
    print(fy,rnd,'areas',W[W.area!='9999'].area.nunique(),'strata',len(W),'suppressed',int(W[['yes','no']].isna().sum().sum()),
          'respondents',int((W.yes+W.no).sum()),'unassigned%',round(100*(W[W.area=='9999'].yes+W[W.area=='9999'].no).sum()/(W.yes+W.no).sum(),2))
    out.append(W)
A=pd.concat(out); A.to_csv(f'{O}/ndb_q20_FY2018-2023_area_sex_age.csv',index=False,encoding='utf-8-sig')
# area code / name consistency across years
nm=A[A.area!='9999'].groupby(['area','fy']).area_nm.first().unstack()
print('areas present in all years:',nm.notna().all(axis=1).sum(),' of',len(nm))
chg=nm[nm.nunique(axis=1)>1]; print('areas whose name differs between years:',len(chg)); print(chg.to_string())
