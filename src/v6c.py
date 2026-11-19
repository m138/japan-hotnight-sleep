exec(open('hn/v6.py').read().split("# --- 図")[0])
for lab,fes in [('M2',['area','yr']),('M4',['area','py'])]:
    P=partial(D,fes); P['year']=D.year.values; P['w']=D.tot.values
    r={}
    for y in range(2018,2024):
        q=P[P.year==y]; m=sm.WLS(q.y,q.tn10u,weights=q.w).fit()
        r[y]=(round(np.exp(m.params.iloc[0]),4), round(float(np.average((q.tn10u*10)**2,weights=q.w)**.5),1))
    print(lab,'年度ごとの傾き(OR/10夜, 調整後熱帯夜の加重SD)',r)
    # 寄与：各年度の Σw x y / Σw x²
    tot=(P.w*P.tn10u*P.y).sum(); den=(P.w*P.tn10u**2).sum()
    print(lab,'分子の年度別シェア',{y:round(float((P.w*P.tn10u*P.y)[P.year==y].sum()/tot),2) for y in range(2018,2024)},' 分母シェア',{y:round(float((P.w*P.tn10u**2)[P.year==y].sum()/den),2) for y in range(2018,2024)})
