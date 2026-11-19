src=open('hn/panel_v5.py').read().split("OUT=[]")[0]; exec(src)
import io,contextlib
R=[]
for v in ['ldens_z','p65_z','single_hh_z','Lmean_z','ac_low']:
    D['ix']=D.tn10u*D[v]
    for lab,fes in [('M2',['area','yr']),('M4',['area','py'])]:
        with contextlib.redirect_stdout(io.StringIO()): fe(D,['tn10u','ix'],fes,YD,f'{lab} {v}',out=R)
for lab,sub in [('AC>=85',D[D.ac_low==0]),('AC<85',D[D.ac_low==1])]:
    with contextlib.redirect_stdout(io.StringIO()): fe(sub,['tn10u'],['area','yr'],YD,f'M2 {lab} n_area={sub.area.nunique()}',out=R)
    print(lab,'mean tn',round(np.average(sub.tn10u*10,weights=sub.tot),1))
R=pd.DataFrame(R); print(R.to_string()); R.to_csv('hn/em_v7.csv',index=False)
