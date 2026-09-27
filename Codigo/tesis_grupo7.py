# -*- coding: utf-8 -*-
"""
===============================================================================
ARQUITECTURA HIBRIDA K-MEANS + RANDOM FOREST / XGBOOST
Pronostico de demanda en retail — Grupo 7 (Melo, Ojeda, Yucasi)
Maestria en Ciencia de Datos — Universidad Continental
===============================================================================

Reproduce los objetivos especificos 1, 2 y 3 sobre los CUATRO conjuntos de
evaluacion, provenientes de TRES fuentes independientes:

    python tesis_grupo7.py privado          # empresa del sector retail peruano
    python tesis_grupo7.py m5               # Walmart M5
    python tesis_grupo7.py rohlik_cz        # Rohlik Group — mercado de Chequia
    python tesis_grupo7.py rohlik_hu        # Rohlik Group — mercado de Hungria

Chequia y Hungria son mercados independientes de la misma fuente: no comparten
ningun producto, operan con estructuras de almacen distintas y en monedas
distintas. En ningun calculo se agregan entre si.

-------------------------------------------------------------------------------
EMBUDO DE LA MUESTRA (identico en los cuatro conjuntos)
-------------------------------------------------------------------------------
1. Patron de demanda: suave + erratico segun Syntetos, Boylan y Croston (2005),
   con los umbrales ADI = 1.32 y CV2 = 0.49 sobre la ventana de entrenamiento.
2. Pronosticabilidad: coeficiente de variacion de los periodos con venta del
   ultimo ciclo estacional menor o igual a 1.00. Umbral tomado de la variable
   PREDECIBLE del sistema de planificacion en produccion de la empresa,
   documentado con anterioridad a esta investigacion.
3. Actividad reciente: venta en al menos 6 periodos del ultimo ciclo y en al
   menos 4 de los ultimos 6 periodos. Umbrales VENDE_6M y FREC_ULT6 del mismo
   sistema en produccion.
4. Precio unitario valido, requisito para las metricas valorizadas.

Ninguno de los criterios utiliza informacion del conjunto de prueba.

ELEGIBILIDAD SIN FUGA DE INFORMACION
La elegibilidad y la muestra se definen EXCLUSIVAMENTE con informacion disponible
al cierre de la ventana de entrenamiento. Un SKU que satisface los criterios al
cierre del entrenamiento permanece en la evaluacion aunque registre venta cero
durante los doce periodos de prueba, y solo puede excluirse mediante informacion
anterior a ese corte. El criterio "registrar venta durante el periodo de
evaluacion", presente en el proyecto de investigacion, queda eliminado: su
justificacion -que el error fuera calculable- no se sostiene, porque el WMAPE
valorizado es un cociente agregado por cluster cuyo denominador es la venta real
valorizada de todo el grupo y no la de cada SKU. La inactividad se detecta con
los criterios de actividad reciente calculados sobre entrenamiento.

Sustento de los criterios 2 y 3:
  Syntetos, Boylan y Croston (2005) clasifica por frecuencia (ADI) y por
  variabilidad del volumen (CV2), pero sus cortes se derivaron para elegir
  entre estimadores, no para decidir pronosticabilidad (Kostenko y Hyndman,
  2006). Boylan, Syntetos y Karakostas (2008) documentan la necesidad de
  revisar esos cortes contra el caso real. Teunter, Syntetos y Babai (2011)
  muestran que los metodos clasicos no detectan obsolescencia porque no
  actualizan tras periodos de demanda nula, lo que produce sesgo positivo.
  Brown (1959) fundamenta el principio de recencia. Bauer (2020) es el
  precedente de delimitar la muestra con criterio de planificacion.

-------------------------------------------------------------------------------
PROTOCOLO
-------------------------------------------------------------------------------
Agrupamiento   : K-means sobre 5 variables de perfil estandarizadas
                 (CV del ciclo, intensidad estacional, variacion interanual,
                 logaritmo del valor y autocorrelacion de orden 1).
Seleccion de K : mayor indice de Rand ajustado sobre 20 remuestreos del 80 %.
                 Empates por debajo de 0.01 se resuelven por indice de silueta.
                 PISO DE NEGOCIO: K >= 3. La segmentacion debe entregar al menos
                 tres grupos con modelo asignado, porque el sistema en
                 produccion mantiene cuatro clusters de los cuales tres reciben
                 modelo de pronostico y el cuarto se gobierna por cuota
                 comercial. Una particion con menos de tres grupos no permite la
                 asignacion diferenciada que exige el objetivo especifico 2.
                 Los tres criterios son anteriores al conjunto de prueba.
Torneo         : Random Forest contra XGBoost por cluster, decidido por WMAPE
                 valorizado sobre la ventana de validacion (ultimos 12 periodos
                 de la ventana de entrenamiento).
Lineas base    : descomposicion clasica multiplicativa (promedio movil) y
                 Holt-Winters aditivo con estacionalidad.
                 El factor de crecimiento del promedio movil se calcula sobre
                 ANIOS CIVILES CERRADOS (enero a diciembre) contenidos en la
                 ventana de entrenamiento, no sobre ventanas moviles de doce
                 periodos contadas desde el inicio de la serie. Es la logica del
                 planificador en produccion y se aplica de forma identica en los
                 cuatro conjuntos.
Peso economico : el valor de cada cluster se calcula EXCLUSIVAMENTE sobre la
                 ventana de entrenamiento. El peso determina cual es el cluster
                 de mayor valor, que es el objeto de la hipotesis especifica 3;
                 calcularlo sobre la serie completa introduciria informacion del
                 periodo de prueba en la definicion del objeto evaluado.
Evaluacion     : WMAPE y BIAS valorizados y en unidades, R2, RMSE y contraste
                 de Wilcoxon bilateral sobre el error absoluto valorizado por
                 producto. El WMAPE valorizado es el indicador de contraste de
                 la hipotesis; el BIAS valorizado se reporta como diagnostico
                 complementario y no se somete a contraste.

-------------------------------------------------------------------------------
DIFERENCIAS DECLARADAS ENTRE CONJUNTOS
-------------------------------------------------------------------------------
Todos los conjuntos se homologan a frecuencia mensual; el ciclo estacional es de
doce periodos en los cuatro.
Meses parciales descartados: M5 descarta 2011-01 (3 dias) y 2016-05 (22 dias);
Rohlik descarta 2024-06 (2 dias, 7.5 % de un mes tipico en Chequia y 7.8 % en
Hungria). Misma regla en ambos casos.
M5 y Rohlik no registran indicador de liquidacion; ese criterio del protocolo se
declara no aplicable en ambos.
El precio se emplea unicamente para valorizar las metricas y para construir la
variable LOG_VALOR. En ningun conjunto ingresa como predictor, de modo que la
informacion de precio, descuento y disponibilidad que publican M5 y Rohlik no se
traslada a la comparacion.

-------------------------------------------------------------------------------
INSUMOS
-------------------------------------------------------------------------------
privado    : matriz_maestra_full_T.xlsx
             (el bloque de objetivo especifico 1 produce resultados_objetivo1.csv)
m5         : m5_mensual_item.csv, m5_precios_item.csv
             (el bloque de objetivo especifico 1 produce m5_resultados_objetivo1.csv,
              m5_matriz.npy y m5_meses.csv)
rohlik_cz  : rohlik_cz_mensual.csv, rohlik_cz_precios.csv
rohlik_hu  : rohlik_hu_mensual.csv, rohlik_hu_precios.csv
             (paneles mensuales por product_unique_id y mercado, obtenidos de
              sales_train.csv e inventory.csv de la competencia publica de
              Rohlik Group; el bloque de objetivo especifico 1 los procesa)

Seed fija en 75 en todo el flujo.
===============================================================================
"""
import sys

_USO = "uso: python tesis_grupo7.py [privado|m5|rohlik_cz|rohlik_hu]"
_VALIDOS = ('privado', 'm5', 'rohlik_cz', 'rohlik_hu')
if len(sys.argv) != 2 or sys.argv[1] not in _VALIDOS:
    print(_USO); sys.exit(1)
_CONJUNTO = sys.argv[1]

# ===========================================================================
# LOCALIZACION DE LOS INSUMOS
# El script no exige que los archivos esten en el directorio de trabajo: los
# busca en las carpetas declaradas abajo, en orden. En Google Colab basta con
# montar la unidad; en un equipo local, agregar la carpeta correspondiente.
# ===========================================================================
import os as _os

CARPETAS = [
    '',                                  # directorio de trabajo actual
    '/content/drive/MyDrive/',           # raiz de la unidad: matriz_maestra_full_T.xlsx
    '/content/drive/MyDrive/M5/',        # m5_mensual_item.csv, m5_precios_item.csv
    '/content/drive/MyDrive/rohlik/',    # rohlik_{cz,hu}_{mensual,precios}.csv
    '/content/drive/MyDrive/tesis_grupo7/',
]

def ruta(nombre):
    """Devuelve la primera ruta existente para 'nombre'; si no lo halla, aborta
    indicando donde busco, en lugar de fallar con un error de pandas."""
    for c in CARPETAS:
        if _os.path.exists(c + nombre):
            return c + nombre
    print(f"NO ENCUENTRO '{nombre}'. Carpetas revisadas:")
    for c in CARPETAS:
        print(f"   {c or '(directorio actual)'}")
    sys.exit(1)


# ===========================================================================
# CONJUNTO 1 — EMPRESA DEL SECTOR RETAIL PERUANO, OBJETIVO ESPECIFICO 1
# ===========================================================================
def privado_oe1(_MERCADO=None):
    # =============================================================================
    # EDA — OBJETIVO ESPECIFICO 1
    # Analisis exploratorio de la estructura de la demanda
    # Universo: los 2,109 SKU del portafolio (filas ">> TOTAL SKU")
    # Ventana de entrenamiento: ENE_2022 -> JUN_2025 (42 meses)
    # Ventana de evaluacion  : JUL_2025 -> JUN_2026 (12 meses)
    # =============================================================================

    import pandas as pd
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore")

    RUTA = ruta('matriz_maestra_full_T.xlsx')

    MESES = ["ENE","FEB","MAR","ABR","MAY","JUN","JUL","AGO","SET","OCT","NOV","DIC"]
    CORTE_TRAIN = 'JUN_2025'          # ultimo mes de entrenamiento

    # -----------------------------------------------------------------------------
    # BLOQUE 0 — CARGA Y ESTRUCTURA
    # -----------------------------------------------------------------------------
    df = pd.read_excel(RUTA)
    df.columns = [str(c).strip() for c in df.columns]

    cols_mes = [f"{m}_{a}" for a in range(2022, 2027) for m in MESES
                if f"{m}_{a}" in df.columns]
    i_corte  = cols_mes.index(CORTE_TRAIN) + 1
    col_train = cols_mes[:i_corte]
    col_test  = cols_mes[i_corte:]

    # Solo la fila de total por SKU: la unidad de analisis es el producto
    es_padre = df['Mercado'].astype(str).str.strip() == '>> TOTAL SKU'
    p = df[es_padre].copy().reset_index(drop=True)

    print("="*72)
    print("BLOQUE 0 — ESTRUCTURA")
    print("="*72)
    print(f"Archivo completo   : {len(df):,} filas x {len(df.columns)} columnas")
    print(f"Unidad de analisis : {len(p):,} SKU (fila total por producto)")
    print(f"Periodos           : {cols_mes[0]} -> {cols_mes[-1]} ({len(cols_mes)} meses)")
    print(f"Entrenamiento      : {col_train[0]} -> {col_train[-1]} ({len(col_train)} meses)")
    print(f"Evaluacion         : {col_test[0]} -> {col_test[-1]} ({len(col_test)} meses)")

    # Calidad de los datos
    V = p[cols_mes].apply(pd.to_numeric, errors='coerce')
    print(f"\nCeldas totales     : {V.size:,}")
    print(f"Celdas vacias      : {int(V.isna().sum().sum()):,}")
    print(f"Celdas negativas   : {int((V < 0).sum().sum()):,}  (devoluciones netas)")
    print(f"SKU duplicados     : {int(p['SKU_Final'].astype(str).str.strip().duplicated().sum())}")
    print(f"Precio no valido   : {int((pd.to_numeric(p['PrecioUnitario'], errors='coerce').fillna(0) <= 0).sum())} SKU")

    # Control padre/hijo declarado en el PAC
    hijos = df[~es_padre].groupby(df.loc[~es_padre, 'SKU_Final'].astype(str).str.strip())[cols_mes].sum()
    tot_p = p.set_index(p['SKU_Final'].astype(str).str.strip())[cols_mes]
    dif = (hijos - tot_p.reindex(hijos.index)).abs().sum().sum()
    print(f"Consistencia total vs suma de mercados: diferencia = {dif:,.0f}")

    # -----------------------------------------------------------------------------
    # BLOQUE 1 — NETEO DE DEVOLUCIONES (hacia atras, ventana 3 meses)
    # El saldo negativo se descuenta del mes de venta positiva mas reciente
    # -----------------------------------------------------------------------------
    M = V.fillna(0).values.astype(float)
    neteadas = 0
    for r in range(M.shape[0]):
        for t in range(M.shape[1]):
            if M[r, t] < 0:
                resto = -M[r, t]
                for k in range(t-1, max(t-4, -1), -1):
                    if M[r, k] > 0:
                        q = min(resto, M[r, k])
                        M[r, k] -= q
                        resto   -= q
                        if resto <= 0:
                            break
                M[r, t] = -resto if resto > 0 else 0
                neteadas += 1

    print(f"\nCeldas negativas tratadas : {neteadas:,}")
    print(f"Negativos remanentes      : {int((M < 0).sum()):,}")
    M = np.clip(M, 0, None)

    precio = pd.to_numeric(p['PrecioUnitario'], errors='coerce').fillna(0).values
    abc    = p['ABC FIN'].astype(str).str.strip().str.upper().values
    sku    = p['SKU_Final'].astype(str).str.strip().values
    n_tr   = len(col_train)

    # -----------------------------------------------------------------------------
    # BLOQUE 2 — CONCENTRACION (participacion acumulada de la venta)
    # -----------------------------------------------------------------------------
    valor = M.sum(axis=1) * precio
    orden = np.argsort(-valor)
    acum  = np.cumsum(valor[orden]) / valor.sum()

    print("\n" + "="*72)
    print("BLOQUE 2 — CONCENTRACION")
    print("="*72)
    for umbral in [0.50, 0.80, 0.95]:
        n = int((acum <= umbral).sum()) + 1
        print(f"  {umbral:.0%} del valor lo explican {n:>5,} SKU  ({n/len(p):>5.1%} del portafolio)")
    print(f"  SKU sin venta valorizada         : {int((valor == 0).sum()):,}")

    # -----------------------------------------------------------------------------
    # BLOQUE 3 — INTERMITENCIA Y VARIABILIDAD (ventana de entrenamiento)
    # ADI = periodos / periodos con venta        (Syntetos et al., 2005)
    # CV2 = (desv/media)^2 sobre periodos con venta
    # -----------------------------------------------------------------------------
    T = M[:, :n_tr]
    adi, cv2, cv, nz_n, estac, autoc = [], [], [], [], [], []

    for r in range(T.shape[0]):
        h  = T[r]
        nz = h[h > 0]
        nz_n.append(len(nz))
        # Autocorrelacion de orden 1: que tan confiable es el mes anterior
        # como indicio del siguiente. Un numero por SKU, describe a la serie.
        autoc.append(pd.Series(h).autocorr(lag=1) if h.std() > 0 else 0.0)
        if len(nz) < 2 or nz.mean() == 0:
            adi.append(np.nan); cv2.append(np.nan); cv.append(np.nan); estac.append(np.nan)
            continue
        adi.append(len(h) / len(nz))
        c = nz.std() / nz.mean()
        cv.append(c); cv2.append(c**2)
        # Intensidad estacional: dispersion del indice mensual sobre 3 anios completos
        m36 = h[:36].reshape(3, 12) if len(h) >= 36 else None
        if m36 is not None and m36.mean() > 0:
            idx = m36.mean(axis=0) / m36.mean()
            estac.append(idx.std())
        else:
            estac.append(np.nan)

    e = pd.DataFrame({'SKU': sku, 'ABC': abc, 'PRECIO': precio, 'VALOR': valor,
                      'MESES_VENTA': nz_n, 'ADI': adi, 'CV': cv, 'CV2': cv2,
                      'ESTAC': estac, 'AUTOCORR': np.nan_to_num(autoc)})

    print("\n" + "="*72)
    print("BLOQUE 3 — INTERMITENCIA Y VARIABILIDAD (42 meses de entrenamiento)")
    print("="*72)
    print(f"  SKU sin historia suficiente (<2 meses con venta): {int(e['ADI'].isna().sum()):,}")
    print(f"\n  Meses con venta  mediana {e['MESES_VENTA'].median():>6.0f} de 42 | "
          f"p25 {e['MESES_VENTA'].quantile(.25):.0f} | p75 {e['MESES_VENTA'].quantile(.75):.0f}")
    print(f"  ADI              mediana {e['ADI'].median():>6.2f} | "
          f"p25 {e['ADI'].quantile(.25):.2f} | p75 {e['ADI'].quantile(.75):.2f}")
    print(f"  CV               mediana {e['CV'].median():>6.2f} | "
          f"p25 {e['CV'].quantile(.25):.2f} | p75 {e['CV'].quantile(.75):.2f}")
    print(f"  Intensidad estac mediana {e['ESTAC'].median():>6.2f} | "
          f"p75 {e['ESTAC'].quantile(.75):.2f} | p90 {e['ESTAC'].quantile(.90):.2f}")

    # -----------------------------------------------------------------------------
    # BLOQUE 4 — CLASIFICACION SYNTETOS-BOYLAN-CROSTON sobre TODO el portafolio
    # -----------------------------------------------------------------------------
    def patron(a, c):
        if np.isnan(a) or np.isnan(c):
            return 'SIN DATOS'
        if a < 1.32 and c < 0.49:  return 'SUAVE'
        if a < 1.32 and c >= 0.49: return 'ERRATICO'
        if a >= 1.32 and c < 0.49: return 'INTERMITENTE'
        return 'CON PICOS'

    e['PATRON'] = [patron(a, c) for a, c in zip(e['ADI'], e['CV2'])]

    print("\n" + "="*72)
    print("BLOQUE 4 — PATRONES DE DEMANDA (Syntetos et al., 2005) | 2,109 SKU")
    print("="*72)
    print(f"  {'PATRON':<14} {'SKU':>7} {'% SKU':>8} {'VALOR':>16} {'% VALOR':>9}")
    print("  " + "-"*58)
    tot_v = e['VALOR'].sum()
    for t in ['SUAVE','ERRATICO','INTERMITENTE','CON PICOS','SIN DATOS']:
        d = e[e['PATRON'] == t]
        if len(d) == 0: continue
        print(f"  {t:<14} {len(d):>7,} {len(d)/len(e):>7.1%} "
              f"{d['VALOR'].sum():>16,.0f} {d['VALOR'].sum()/tot_v:>8.1%}")

    # -----------------------------------------------------------------------------
    # BLOQUE 5 — CRUCE ABC vigente x PATRON  (el hallazgo que sustenta el problema)
    # -----------------------------------------------------------------------------
    print("\n" + "="*72)
    print("BLOQUE 5 — ABC VIGENTE DE LA EMPRESA vs PATRON DE DEMANDA")
    print("="*72)
    ct = pd.crosstab(e['ABC'], e['PATRON'])
    print(ct.to_string())

    regulares = e['PATRON'].isin(['SUAVE','ERRATICO'])
    a_fuera = e[(e['ABC'] == 'A') & (~regulares)]
    print(f"\n  SKU clase A que NO son pronosticables por metodos regulares: {len(a_fuera):,}")
    print(f"  Valor que representan: {a_fuera['VALOR'].sum():,.0f} "
          f"({a_fuera['VALOR'].sum()/tot_v:.1%} del portafolio)")

    # -----------------------------------------------------------------------------
    # BLOQUE 6 — EMBUDO DE ELEGIBILIDAD DECLARADO EN EL PAC
    # -----------------------------------------------------------------------------
    print("\n" + "="*72)
    print("BLOQUE 6 — EMBUDO DE ELEGIBILIDAD")
    print("="*72)

    # ============================================================================
    # La elegibilidad se define EXCLUSIVAMENTE con informacion disponible al cierre
    # de la ventana de entrenamiento. El criterio "registrar venta durante el periodo
    # de evaluacion" queda eliminado: consultar el conjunto de prueba para decidir
    # quien entra a la muestra introduce informacion del futuro en la definicion del
    # objeto evaluado. Un SKU elegible al cierre del TRAIN permanece en la evaluacion
    # aunque venda cero durante los doce meses de prueba.
    # La justificacion original de ese criterio -que el error fuera calculable- no se
    # sostiene: el WMAPE valorizado es un cociente agregado por grupo, cuyo
    # denominador es la venta real valorizada de todo el grupo y no la de cada SKU.
    # La inactividad se detecta con los criterios de actividad reciente calculados
    # sobre TRAIN (venta en >= 6 de los ultimos 12 periodos y en >= 4 de los ultimos 6).
    # ============================================================================
    vivo_train = np.clip(M[:, :n_tr], 0, None).sum(axis=1) > 0   # venta en entrenamiento
    hist_24   = np.array([np.flatnonzero(M[r, :n_tr] > 0)[0] <= n_tr - 24
                          if (M[r, :n_tr] > 0).any() else False
                          for r in range(M.shape[0])])           # >=24 meses de historia
    remate    = p['Remate'].astype(str).str.strip().str.upper().ne('N').values   # 'N' = no remate
    precio_ok = precio > 0

    paso = pd.DataFrame({'crit': ['Poblacion',
                                  'Sin venta en la ventana de entrenamiento',
                                  'Historial menor a 24 meses',
                                  'Liquidacion o remate',
                                  'Precio unitario no valido'],
                         'quedan': [len(e),
                                    int(vivo_train.sum()),
                                    int((vivo_train & hist_24).sum()),
                                    int((vivo_train & hist_24 & ~remate).sum()),
                                    int((vivo_train & hist_24 & ~remate & precio_ok).sum())]})
    prev = len(e)
    for _, r in paso.iterrows():
        desc = prev - r['quedan']
        print(f"  {r['crit']:<42} descartan {desc:>5,} | quedan {r['quedan']:>6,}")
        prev = r['quedan']

    elegible = vivo_train & hist_24 & ~remate & precio_ok
    _sin_test = int((elegible & regulares.values & (M[:, n_tr:].sum(axis=1) <= 0)).sum())
    print(f"  De la muestra, SKU sin venta en evaluacion: {_sin_test} "
          f"(se conservan; el conjunto de prueba solo se usa para evaluar)")
    print(f"\n  POBLACION DE ESTUDIO : {int(elegible.sum()):,} SKU | "
          f"{e.loc[elegible,'VALOR'].sum()/tot_v:.1%} del valor")

    muestra = elegible & regulares.values
    print(f"  MUESTRA (suave+erratico): {int(muestra.sum()):,} SKU | "
          f"{e.loc[muestra,'VALOR'].sum()/tot_v:.1%} del valor")
    print(f"  Descartados por patron  : {int((elegible & ~regulares.values).sum()):,} SKU")

    e['ELEGIBLE'] = elegible
    e['MUESTRA']  = muestra

    # -----------------------------------------------------------------------------
    # BLOQUE 7 — VARIABLES PREDICTORAS Y MULTICOLINEALIDAD (sobre la muestra)
    # -----------------------------------------------------------------------------
    idx = np.flatnonzero(muestra)
    F = []
    for r in idx:
        h = M[r, :n_tr]
        nz = h[h > 0]
        ult12, prev12 = h[-12:], h[-24:-12]
        F.append({
            'LAG_1'      : h[-1],
            'LAG_12'     : h[-12],
            'MEDIA_3'    : h[-3:].mean(),
            'MEDIA_12'   : ult12.mean(),
            'FREC_12'    : int((ult12 > 0).sum()),
            'CV_12'      : ult12.std() / ult12.mean() if ult12.mean() > 0 else 0,
            'YOY'        : (ult12.sum()/prev12.sum() - 1) if prev12.sum() > 0 else 0,
            'LOG_VALOR'  : np.log1p(ult12.mean() * precio[r]),
            'ESTAC'      : e['ESTAC'].iloc[r] if not np.isnan(e['ESTAC'].iloc[r]) else 0,
            'AUTOCORR'   : e['AUTOCORR'].iloc[r],
        })
    F = pd.DataFrame(F).replace([np.inf, -np.inf], 0).fillna(0)

    print("\n" + "="*72)
    print(f"BLOQUE 7 — VARIABLES PREDICTORAS | n = {len(F):,} SKU de la muestra")
    print("="*72)
    print("\n  Matriz de correlacion (Pearson):")
    print(F.corr().round(2).to_string())

    # Factor de inflacion de la varianza
    from sklearn.linear_model import LinearRegression
    print(f"\n  {'VARIABLE':<12} {'VIF':>8}")
    print("  " + "-"*22)
    Xs = (F - F.mean()) / F.std().replace(0, 1)
    for c in F.columns:
        y = Xs[c].values
        X = Xs.drop(columns=[c]).values
        r2 = LinearRegression().fit(X, y).score(X, y)
        vif = 1/(1-r2) if r2 < 0.9999 else np.inf
        print(f"  {c:<12} {vif:>8.2f}")

    e.to_csv('eda_oe1_resultados.csv', index=False)
    # Archivo de entrada de los objetivos especificos 2 y 3 del conjunto privado.
    e.to_csv('resultados_objetivo1.csv', index=False)
    print("\n  Detalle por SKU guardado en eda_oe1_resultados.csv")


# ===========================================================================
# CONJUNTO 1 — EMPRESA DEL SECTOR RETAIL PERUANO, OBJETIVOS ESPECIFICOS 2 y 3
# ===========================================================================
def correr_privado(_MERCADO=None):
    # =============================================================================
    # TESIS — CORRIDA FINAL K = 3, VALIDACION SIN FUGA
    # Genera el perfil de los grupos y las seis figuras del Capitulo V
    # =============================================================================
    import os, warnings, itertools
    warnings.filterwarnings("ignore")
    import pandas as pd, numpy as np
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                                 calinski_harabasz_score, davies_bouldin_score,
                                 mean_squared_error, r2_score)
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    from scipy.stats import wilcoxon

    SEED, K = 75, 3
    # Piso de negocio: la segmentacion debe entregar al menos tres grupos con modelo.
    # Anclaje en produccion: el simulador de la empresa mantiene cuatro clusters, de
    # los cuales tres reciben modelo de pronostico y el cuarto se gobierna por cuota
    # comercial. Este conjunto opera con K = 3 y satisface el piso.
    K_MIN = 3
    assert K >= K_MIN, "la particion no alcanza el piso de negocio de tres grupos"
    RUTA  = ruta('matriz_maestra_full_T.xlsx')
    MESES = ["ENE","FEB","MAR","ABR","MAY","JUN","JUL","AGO","SET","OCT","NOV","DIC"]
    SAL = 'figuras_v6'; os.makedirs(SAL, exist_ok=True)

    COL = {'PM':'#2F6F9F','HW':'#C56B2C','ML':'#12876A'}
    CL_COL = ['#2F6F9F','#C56B2C','#12876A','#7D6A9C']
    TINTA, TINTA2, REJ = '#2b2b2b', '#5c5c5c', '#d8d8d4'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.edgecolor':REJ,
     'axes.linewidth':0.8,'axes.labelcolor':TINTA,'text.color':TINTA,'xtick.color':TINTA2,
     'ytick.color':TINTA2,'figure.facecolor':'white','axes.facecolor':'white',
     'savefig.dpi':300,'savefig.bbox':'tight'})
    def limpiar(ax):
        for l in ['top','right']: ax.spines[l].set_visible(False)
        ax.grid(axis='y', color=REJ, linewidth=0.6, alpha=0.8); ax.set_axisbelow(True)

    # ---------------- carga ----------------
    df = pd.read_excel(RUTA); df.columns = [str(c).strip() for c in df.columns]
    cols = [f"{m}_{a}" for a in range(2022,2027) for m in MESES if f"{m}_{a}" in df.columns]
    n_tr = cols.index('JUN_2025') + 1; H = len(cols) - n_tr
    p = df[df['Mercado'].astype(str).str.strip()=='>> TOTAL SKU'].reset_index(drop=True)
    M = p[cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
    for r in range(M.shape[0]):
        for t in range(M.shape[1]):
            if M[r,t] < 0:
                resto = -M[r,t]
                for k in range(t-1, max(t-4,-1), -1):
                    if M[r,k] > 0:
                        q = min(resto, M[r,k]); M[r,k] -= q; resto -= q
                        if resto <= 0: break
                M[r,t] = -resto if resto > 0 else 0.0
    Mpos = np.clip(M, 0, None)
    precio = pd.to_numeric(p['PrecioUnitario'], errors='coerce').fillna(0).values
    E = pd.read_csv('resultados_objetivo1.csv'); idx = np.flatnonzero(E['MUESTRA'].values)

    # ============================================================================
    # CRITERIOS DE NEGOCIO EN EL EMBUDO (ambos anteriores al conjunto de prueba)
    #  A) PRONOSTICABILIDAD  : CV de los periodos con venta del ultimo ciclo <= 1.0
    #     Umbral PREDECIBLE del simulador en produccion. Ventana: un ciclo estacional
    #     (12 meses en los cuatro conjuntos evaluados), por coherencia con LAG_12, MEDIA_12,
    #     FREC_12 y el horizonte de prueba. En produccion la ventana es de 24 meses
    #     porque alli responde a la politica de reposicion, no al ciclo de pronostico.
    #  B) ACTIVIDAD RECIENTE : venta en >= 6 periodos del ultimo ciclo Y en >= 4 de
    #     los ultimos 6 periodos (VENDE_6M y FREC_ULT6 del simulador).
    # Sustento: Brown (1959) recencia; Syntetos et al. (2005) patron; Boylan,
    # Syntetos y Karakostas (2008) revision de cortes; Teunter, Syntetos y Babai
    # (2011) obsolescencia; Bauer (2020) delimitacion por criterio de planificacion.
    # ============================================================================
    def _cv_ciclo(h):
        v = h[-12:]; nz = v[v > 0]
        return nz.std(ddof=1)/nz.mean() if len(nz) >= 2 and nz.mean() > 0 else 0.0
    _cv  = np.array([_cv_ciclo(Mpos[r, :n_tr]) for r in idx])
    _f12 = np.array([(Mpos[r, n_tr-12:n_tr] > 0).sum() for r in idx])
    _u6  = np.array([(Mpos[r, n_tr-6:n_tr]    > 0).sum() for r in idx])
    _pred = _cv <= 1.0
    _act  = (_f12 >= 6) & (_u6 >= 4)
    print(f"EMBUDO DE NEGOCIO | no pronosticables (CV>1): {int((~_pred).sum()):,} | "
          f"sin actividad reciente: {int((_pred & ~_act).sum()):,} | "
          f"quedan {int((_pred & _act).sum()):,} de {len(idx):,}", flush=True)
    idx = idx[_pred & _act]

    val_sku = Mpos[idx][:, :n_tr].sum(axis=1)*precio[idx]

    def f_est(y):
        if len(y) < 24 or np.std(y) == 0: return 0.0
        try:
            d = seasonal_decompose(pd.Series(y), model='additive', period=12, extrapolate_trend='freq')
            r_, s_ = d.resid.dropna(), None
            s_ = d.seasonal[r_.index]; v = np.var(r_+s_)
            return float(max(0.0, 1-np.var(r_)/v)) if v > 0 else 0.0
        except Exception: return 0.0

    def perfil(h, t, pu):
        v12 = h[t-12:t]; pv = h[t-24:t-12] if t >= 24 else np.array([0.0])
        return {'FREC_12': float((v12>0).sum()),
                'CV_12': v12.std()/v12.mean() if v12.mean()>0 else 0.0,
                'ESTAC': f_est(h[:t]),
                'YOY': (v12.sum()/pv.sum()-1) if pv.sum()>0 else 0.0,
                'LOG_VALOR': float(np.log1p(v12.mean()*pu)),
                'AUTOCORR': float(pd.Series(h[:t]).autocorr(lag=1)) if h[:t].std()>0 else 0.0}

    print("Perfil movil...")
    MOVIL = {j: {t: perfil(Mpos[r,:n_tr], t, precio[r]) for t in range(12, n_tr)}
             for j, r in enumerate(idx)}
    G0 = pd.DataFrame([perfil(Mpos[r,:n_tr], n_tr, precio[r]) for r in idx])
    G0 = G0.replace([np.inf,-np.inf],0).fillna(0)
    G0['YOY'] = G0['YOY'].clip(G0['YOY'].quantile(.01), G0['YOY'].quantile(.99))
    VARS_CL = ['CV_12','ESTAC','YOY','LOG_VALOR','AUTOCORR']
    X = StandardScaler().fit_transform(G0[VARS_CL])

    # ============ COMPARACION DE ALGORITMOS DE AGRUPAMIENTO ============
    print("\n" + "="*76); print("COMPARACION DE ALGORITMOS DE AGRUPAMIENTO (K = 3)"); print("="*76)
    print(f"{'ALGORITMO':<28} {'SILUETA':>9} {'CALINSKI':>10} {'DAVIES':>9} {'RAND':>8}")
    print("-"*76)
    algos = {
     'K-means': KMeans(n_clusters=K, random_state=SEED, n_init=20),
     'Jerárquico (Ward)': AgglomerativeClustering(n_clusters=K, linkage='ward'),
     'Jerárquico (promedio)': AgglomerativeClustering(n_clusters=K, linkage='average'),
     'Mezclas gaussianas': GaussianMixture(n_components=K, random_state=SEED, n_init=10),
    }
    comp_algo = {}
    for nom, mdl in algos.items():
        lb = mdl.fit_predict(X)
        if len(set(lb)) < 2: continue
        rng, ars = np.random.default_rng(SEED), []
        for _ in range(20):
            s = rng.choice(len(X), int(len(X)*0.8), replace=False)
            try:
                m2 = type(mdl)(**mdl.get_params()); l2 = m2.fit_predict(X[s])
                ars.append(adjusted_rand_score(lb[s], l2))
            except Exception: pass
        comp_algo[nom] = dict(sil=silhouette_score(X, lb), ch=calinski_harabasz_score(X, lb),
                              db=davies_bouldin_score(X, lb), ari=np.mean(ars) if ars else np.nan,
                              lab=lb)
        print(f"{nom:<28} {comp_algo[nom]['sil']:>9.3f} {comp_algo[nom]['ch']:>10.1f} "
              f"{comp_algo[nom]['db']:>9.3f} {comp_algo[nom]['ari']:>8.3f}")

    lab = comp_algo['K-means']['lab']
    G0['CLUSTER'] = lab

    # ---------------- nombres y orden ----------------
    perf = G0.groupby('CLUSTER')[VARS_CL].mean()
    libres, nombre = list(perf.index), {}
    c = perf.loc[libres,'LOG_VALOR'].idxmax(); nombre[c]='Núcleo de alto valor'; libres.remove(c)
    c = perf.loc[libres,'YOY'].idxmax();       nombre[c]='Emergentes de crecimiento explosivo'; libres.remove(c)
    nombre[libres[0]] = 'Cola volátil de bajo valor'
    ORDEN = sorted(perf.index, key=lambda c: -val_sku[lab == c].sum())
    NUM = {c: i+1 for i, c in enumerate(ORDEN)}

    print("\n" + "="*84); print("PERFIL DE LOS GRUPOS (K = 3)"); print("="*84)
    print(f"{'N':>2} {'GRUPO':<38} {'SKU':>5} {'CV':>6} {'ESTAC':>6} {'YOY':>7} {'AUTOC':>7} {'%VALOR':>8}")
    for c in ORDEN:
        d = G0[G0['CLUSTER']==c]
        print(f"{NUM[c]:>2} {nombre[c]:<38} {len(d):>5} {d['CV_12'].mean():>6.2f} "
              f"{d['ESTAC'].mean():>6.2f} {d['YOY'].mean():>+7.2f} {d['AUTOCORR'].mean():>+7.2f} "
              f"{val_sku[lab==c].sum()/val_sku.sum():>7.1%}")

    # ---------------- torneo y prediccion ----------------
    FIN_VAL = n_tr - 12
    def fila(h, t, d):
        v12 = h[t-12:t]
        return [h[t-1], h[t-12], h[t-3:t].mean(), v12.mean(), (t%12)+1,
                d['FREC_12'], d['CV_12'], d['ESTAC'], d['YOY'], d['LOG_VALOR'], d['AUTOCORR']]
    NOMB_F = ['LAG_1','LAG_12','MEDIA_3','MEDIA_12','MES','FREC_12','CV_12','ESTAC','YOY','LOG_VALOR','AUTOCORR']
    REJ_RF  = [{'n_estimators':400,'max_depth':d,'min_samples_leaf':l} for d,l in itertools.product([8,14,None],[1,3])]
    REJ_XGB = [{'n_estimators':n,'max_depth':d,'learning_rate':lr,'subsample':0.8,
                'colsample_bytree':0.8,'min_child_weight':w}
               for n,d,lr,w in itertools.product([400,800],[4,6],[0.03,0.08],[1,5])]

    print("\n" + "="*84); print("TORNEO POR GRUPO (WMAPE valorizado en validacion)"); print("="*84)
    pred_ml = np.zeros((len(idx), H)); torneo = {}
    for c in ORDEN:
        pos = np.flatnonzero(lab == c)
        Xtr, ytr, Xva, yva, pva = [], [], [], [], []
        for j in pos:
            h = Mpos[idx[j], :n_tr]
            for t in range(12, n_tr):
                f = fila(h, t, MOVIL[j][t])
                if t < FIN_VAL: Xtr.append(f); ytr.append(h[t])
                else: Xva.append(f); yva.append(h[t]); pva.append(precio[idx[j]])
        Xtr, ytr, Xva, yva, pva = map(np.array, (Xtr, ytr, Xva, yva, pva))
        wm = lambda yp: np.abs((yva-yp)*pva).sum()/max((yva*pva).sum(),1)
        b = {}
        for nm, rj, cl in [('RF',REJ_RF,RandomForestRegressor), ('XGB',REJ_XGB,XGBRegressor)]:
            mj, hb = np.inf, None
            for hp in rj:
                w = wm(cl(random_state=SEED, n_jobs=-1, **hp).fit(Xtr, ytr).predict(Xva))
                if w < mj: mj, hb = w, hp
            b[nm] = (mj, hb)
        gan = 'RF' if b['RF'][0] <= b['XGB'][0] else 'XGB'
        torneo[c] = dict(rf=b['RF'][0], xgb=b['XGB'][0], gan=gan, hp=b[gan][1])
        print(f"{NUM[c]:>2} {nombre[c]:<38} RF {b['RF'][0]:>6.1%}  XGB {b['XGB'][0]:>6.1%}  -> {gan}")
        cl = RandomForestRegressor if gan=='RF' else XGBRegressor
        mdl = cl(random_state=SEED, n_jobs=-1, **b[gan][1]).fit(np.vstack([Xtr,Xva]), np.concatenate([ytr,yva]))
        torneo[c]['imp'] = dict(zip(NOMB_F, mdl.feature_importances_))
        hist = [np.array(Mpos[idx[j], :n_tr], dtype=float) for j in pos]
        for t in range(H):
            Xb = np.array([fila(hist[i], len(hist[i]), perfil(hist[i], len(hist[i]), precio[idx[j]]))
                           for i, j in enumerate(pos)])
            for i, v in enumerate(mdl.predict(Xb)): hist[i] = np.append(hist[i], max(0.0, float(v)))
        for i, j in enumerate(pos): pred_ml[j] = hist[i][n_tr:n_tr+H]

    # ---------------- lineas base ----------------

    # ============================================================================
    # LINEA BASE 1 — PROMEDIO MOVIL (descomposicion clasica multiplicativa)
    # Funcion unica empleada en los cuatro conjuntos evaluados.
    #   1. Nivel = promedio del ultimo ciclo estacional.
    #   2. Indice estacional = promedio historico de cada posicion del ciclo
    #      dividido entre el promedio general de la serie.
    #   3. Crecimiento = promedio ponderado linealmente de los ratios interanuales
    #      entre ANIOS CIVILES CERRADOS (enero a diciembre) contenidos en la
    #      ventana de entrenamiento, con mayor peso en los mas recientes. Esta es
    #      la logica de negocio del sistema en produccion: el planificador compara
    #      anios cerrados, no ventanas moviles.
    #   4. El crecimiento se acota al rango [0.5, 1.5].
    #   5. El pronostico final tiene piso en cero.
    # Guardas: nivel no positivo devuelve ceros; posicion estacional con promedio
    # historico nulo recibe indice 1.0 en lugar de 0.0.
    # ============================================================================
    def pm(h, S, H, blq=None):
        nivel = h[-S:].mean()
        if nivel <= 0:
            return np.zeros(H)
        prom = h.mean() if h.mean() > 0 else 1.0
        ind = np.array([h[i::S].mean()/prom if h[i::S].mean() > 0 else 1.0 for i in range(S)])
        if blq:
            an = [h[b].sum() for b in blq]
        else:
            an = [h[i*S:(i+1)*S].sum() for i in range(len(h)//S)]
        rt = [an[i+1]/an[i] for i in range(len(an)-1) if an[i] > 0]
        g = float(np.clip(np.average(rt, weights=range(1, len(rt)+1)), 0.5, 1.5)) if rt else 1.0
        return np.array([max(0.0, nivel*ind[(len(h)+k) % S]*g) for k in range(H)])

    # Anios civiles cerrados (ENE-DIC) contenidos en la ventana de entrenamiento
    _an = {}
    for _i, _c in enumerate(cols[:n_tr]):
        _an.setdefault(_c.split('_')[1], []).append(_i)
    CERRADOS = [a for a in sorted(_an) if len(_an[a]) == 12]
    BLQ = [_an[a] for a in CERRADOS]
    print(f"ANIOS CIVILES CERRADOS EN ENTRENAMIENTO: {CERRADOS} -> "
          f"{max(0,len(CERRADOS)-1)} ratio(s) interanual(es)", flush=True)

    pred_pm, pred_hw, real = (np.zeros_like(pred_ml) for _ in range(3))
    for j, r in enumerate(idx):
        h = Mpos[r,:n_tr]; real[j] = Mpos[r, n_tr:n_tr+H]
        pred_pm[j] = pm(h, 12, H, BLQ)
        try:
            pred_hw[j] = np.clip(ExponentialSmoothing(h, trend='add', seasonal='add', seasonal_periods=12,
                initialization_method='estimated').fit(optimized=True).forecast(H), 0, None)
        except Exception: pred_hw[j] = pred_pm[j]
    pr = precio[idx].reshape(-1,1)

    def met(mask, pred):
        R, P = real[mask]*pr[mask], pred[mask]*pr[mask]; v = R.sum()
        U, Q = real[mask], pred[mask]; u = U.sum()
        return dict(wv=np.abs(R-P).sum()/max(v,1), bv=P.sum()/max(v,1)-1,
                    wu=np.abs(U-Q).sum()/max(u,1), bu=Q.sum()/max(u,1)-1,
                    rmse=np.sqrt(mean_squared_error(U.ravel(), Q.ravel())),
                    r2=r2_score(U.ravel(), Q.ravel()))

    print("\n" + "="*100); print("DESEMPENIO EN EL CONJUNTO DE PRUEBA"); print("="*100)
    print(f"{'N':>2} {'GRUPO':<38} {'METODO':<22} {'WMAPE$':>8} {'BIAS$':>8} {'WMAPEu':>8} {'BIASu':>8} {'RMSE':>8} {'R2':>6}")
    nm_met = {'PM':'Promedio móvil','HW':'Holt-Winters','ML':'Arquitectura híbrida'}
    for c in ORDEN:
        mk = lab == c
        for et, pdd in [('PM',pred_pm),('HW',pred_hw),('ML',pred_ml)]:
            m = met(mk, pdd)
            print(f"{NUM[c]:>2} {nombre[c]:<38} {nm_met[et]:<22} {m['wv']:>7.1%} {m['bv']:>+7.1%} "
                  f"{m['wu']:>7.1%} {m['bu']:>+7.1%} {m['rmse']:>8.1f} {m['r2']:>6.2f}")

    ae = {et:(np.abs(real-pdd)*pr).sum(axis=1) for et,pdd in [('PM',pred_pm),('HW',pred_hw),('ML',pred_ml)]}
    print("\n" + "="*84); print("CONTRASTE DE WILCOXON"); print("="*84)
    for c in ORDEN:
        mk = lab == c; ps = []
        for b_ in ['PM','HW']:
            try: ps.append(f"{wilcoxon(ae[b_][mk], ae['ML'][mk])[1]:.2e}")
            except Exception: ps.append("n/a")
        print(f"{NUM[c]:>2} {nombre[c]:<38} vs PM {ps[0]:>10} vs HW {ps[1]:>10} | "
              f"${ae['PM'][mk].sum()-ae['ML'][mk].sum():>+11,.0f} ${ae['HW'][mk].sum()-ae['ML'][mk].sum():>+11,.0f}")

    print("\nImportancia de variables por grupo (top 5):")
    for c in ORDEN:
        s = pd.Series(torneo[c]['imp']).sort_values(ascending=False).head(5)
        print(f"  Grupo {NUM[c]}: " + " · ".join(f"{k} {v:.3f}" for k, v in s.items()))

    # =============================================================================
    # FIGURAS
    # =============================================================================
    valor_all = Mpos.sum(axis=1)*precio
    acum = np.cumsum(np.sort(valor_all)[::-1])/valor_all.sum()

    # Figura 1 — concentracion
    fig, ax = plt.subplots(figsize=(6.4,3.4))
    x = np.arange(1, len(acum)+1); ax.plot(x, acum*100, color=COL['ML'], linewidth=2)
    n80 = int((acum <= 0.80).sum())+1
    ax.axhline(80, color=REJ, linewidth=1, linestyle='--')
    ax.plot([n80],[80],'o',color=COL['ML'],markersize=8,markeredgecolor='white',markeredgewidth=2)
    ax.annotate(f'{n80} SKU ({n80/len(p):.1%})\nexplican el 80 % del valor', xy=(n80,80),
                xytext=(n80+260,55), color=TINTA, arrowprops=dict(arrowstyle='-',color=TINTA2,linewidth=0.8))
    ax.set_xlabel('SKU ordenados de mayor a menor venta valorizada'); ax.set_ylabel('Valor acumulado (%)')
    ax.set_xlim(0,len(acum)); ax.set_ylim(0,102); limpiar(ax)
    fig.savefig(f'{SAL}/fig1_concentracion.png'); plt.close(fig)

    # Figura 2 — patrones
    fig, ax = plt.subplots(figsize=(6.4,4.2))
    mapa = {'Suave':CL_COL[2],'Erratico':CL_COL[0],'Intermitente':CL_COL[1],'Con picos':CL_COL[3]}
    d = E.dropna(subset=['ADI','CV2'])
    for t_, col in mapa.items():
        s = d[d['PATRON']==t_]
        ax.scatter(s['ADI'], s['CV2'], s=16, color=col, alpha=.65, linewidths=.5,
                   edgecolors='white', label=f'{t_} (n={len(s)})')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.axvline(1.32,color=TINTA2,linewidth=1,linestyle='--'); ax.axhline(0.49,color=TINTA2,linewidth=1,linestyle='--')
    y0,_ = ax.get_ylim(); _,x1 = ax.get_xlim()
    ax.text(1.40, y0*1.6, 'ADI = 1.32', color=TINTA2, fontsize=8)
    ax.text(x1*0.30, 0.56, 'CV² = 0.49', color=TINTA2, fontsize=8)
    ax.set_xlabel('Intervalo medio entre demandas (escala logarítmica)')
    ax.set_ylabel('CV² de las cantidades (escala logarítmica)')
    ax.legend(frameon=False, fontsize=8, ncol=4, loc='upper center', bbox_to_anchor=(0.5,1.14),
              handletextpad=.3, columnspacing=1.2)
    limpiar(ax); ax.grid(axis='x', color=REJ, linewidth=.6, alpha=.8)
    fig.savefig(f'{SAL}/fig2_patrones.png'); plt.close(fig)

    # Figura 3 — matriz de correlacion (pedido del profesor)
    FV = []
    for j, r in enumerate(idx):
        h = Mpos[r,:n_tr]; d_ = MOVIL[j][n_tr-1]
        FV.append(dict(zip(NOMB_F, fila(h, n_tr, perfil(h, n_tr, precio[r])))))
    FV = pd.DataFrame(FV).replace([np.inf,-np.inf],0).fillna(0)
    FV['VENTA'] = real.sum(axis=1)
    Cm = FV.corr()
    fig, ax = plt.subplots(figsize=(6.6,5.4))
    im = ax.imshow(Cm.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(Cm))); ax.set_xticklabels(Cm.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(Cm))); ax.set_yticklabels(Cm.columns, fontsize=8)
    for i in range(len(Cm)):
        for k in range(len(Cm)):
            v = Cm.values[i,k]
            ax.text(k, i, f'{v:.2f}', ha='center', va='center', fontsize=6.5,
                    color='white' if abs(v) > 0.55 else TINTA)
    cb = fig.colorbar(im, ax=ax, shrink=.8); cb.outline.set_visible(False)
    cb.set_label('Coeficiente de correlación de Pearson', fontsize=8)
    for l in ['top','right','bottom','left']: ax.spines[l].set_visible(False)
    ax.set_xticks(np.arange(-.5,len(Cm),1), minor=True); ax.set_yticks(np.arange(-.5,len(Cm),1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5); ax.tick_params(which='minor', length=0)
    fig.savefig(f'{SAL}/fig3_correlacion.png'); plt.close(fig)

    # Figura 4 — bivariados contra la venta (pedido del profesor)
    VB = ['MEDIA_12','LAG_12','CV_12','FREC_12','ESTAC','AUTOCORR']
    fig, axs = plt.subplots(2, 3, figsize=(7.2,4.6))
    for a, v in zip(axs.ravel(), VB):
        a.scatter(FV[v], FV['VENTA'], s=10, color=COL['ML'], alpha=.45, linewidths=0)
        if FV[v].std() > 0:
            z = np.polyfit(FV[v], FV['VENTA'], 1)
            xs = np.linspace(FV[v].min(), FV[v].max(), 50)
            a.plot(xs, np.polyval(z, xs), color=COL['HW'], linewidth=1.5)
        a.set_xlabel(v, fontsize=8); a.set_yscale('symlog')
        a.set_title(f'r = {FV[v].corr(FV["VENTA"]):.2f}', fontsize=8, color=TINTA2)
        limpiar(a); a.tick_params(labelsize=7)
    axs[0,0].set_ylabel('Venta del periodo de prueba', fontsize=8)
    axs[1,0].set_ylabel('Venta del periodo de prueba', fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{SAL}/fig4_bivariados.png'); plt.close(fig)

    # Figura 5 — grupos
    fig, ax = plt.subplots(figsize=(6.4,4.2))
    for k_, c in enumerate(ORDEN):
        m_ = lab == c
        ax.scatter(G0.loc[m_,'CV_12'], G0.loc[m_,'LOG_VALOR'], s=18, color=CL_COL[k_], alpha=.7,
                   linewidths=.5, edgecolors='white', label=f"{NUM[c]}. {nombre[c]} (n={int(m_.sum())})")
    ax.set_xlabel('Coeficiente de variación de los últimos 12 meses')
    ax.set_ylabel('Valor de venta mensual (escala logarítmica)')
    ax.legend(frameon=False, fontsize=8, loc='upper right'); limpiar(ax)
    fig.savefig(f'{SAL}/fig5_grupos.png'); plt.close(fig)

    # Figura 6 — WMAPE por grupo y metodo
    fig, ax = plt.subplots(figsize=(6.8,3.8))
    anc, xs = 0.26, np.arange(len(ORDEN))
    for i, (et, pdd) in enumerate([('PM',pred_pm),('HW',pred_hw),('ML',pred_ml)]):
        vals = [met(lab==c, pdd)['wv']*100 for c in ORDEN]
        bs = ax.bar(xs+(i-1)*anc, vals, anc*0.9, color=COL[et], label=nm_met[et])
        for b_, v in zip(bs, vals):
            ax.text(b_.get_x()+b_.get_width()/2, v+1.5, f'{v:.0f}', ha='center', fontsize=8, color=TINTA)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{NUM[c]}. {nombre[c].split(' ')[0]}\n({val_sku[lab==c].sum()/val_sku.sum():.0%} del valor)"
                        for c in ORDEN], fontsize=8)
    ax.set_ylabel('WMAPE valorizado (%)')
    ax.legend(frameon=False, fontsize=8, ncol=3, loc='upper left'); limpiar(ax)
    fig.savefig(f'{SAL}/fig6_wmape.png'); plt.close(fig)

    # ---------------- guardado ----------------
    res = []
    for c in ORDEN:
        mk = lab == c
        for et, pdd in [('PM',pred_pm),('HW',pred_hw),('ML',pred_ml)]:
            res.append(dict(grupo=NUM[c], nombre=nombre[c], n=int(mk.sum()),
                            peso=val_sku[mk].sum()/val_sku.sum(), metodo=et, **met(mk, pdd)))
    pd.DataFrame(res).to_csv('v6_priv_resultados.csv', index=False)
    print(f"\nFiguras en {SAL}/ y resultados en resultados_tesis_k3.csv")


# ===========================================================================
# CONJUNTO 2 — WALMART M5, OBJETIVO ESPECIFICO 1
# ===========================================================================
def m5_oe1(_MERCADO=None):
    # =============================================================================
    # OBJETIVO ESPECIFICO 1 — M5 (Walmart)
    # Mismo protocolo declarado en el PAC para el conjunto de la empresa.
    #
    # Unidad de analisis: item_id (SKU), agregando las 10 tiendas.
    # Se descartan los dos meses incompletos: 2011-01 (3 dias) y 2016-05 (22 dias).
    # =============================================================================
    import warnings; warnings.filterwarnings("ignore")
    import pandas as pd, numpy as np

    ADI_U, CV2_U = 1.32, 0.49          # umbrales Syntetos et al. (2005)
    H_TEST = 12                        # meses de evaluacion, igual que el privado

    V = pd.read_csv(ruta('m5_mensual_item.csv'))
    P = pd.read_csv(ruta('m5_precios_item.csv'))

    meses_all = [c for c in V.columns if c[:2] == '20']
    meses = [m for m in meses_all if m not in ('2011-01', '2016-05')]   # completos
    n_tr = len(meses) - H_TEST
    col_tr, col_te = meses[:n_tr], meses[n_tr:]

    M = V[meses].values.astype(float)
    item = V['item_id'].values
    dept = V['dept_id'].values
    cat  = V['cat_id'].values
    precio = P.set_index('item_id').loc[item, 'precio_medio'].values

    L = 78
    print("="*L); print("BLOQUE 0 — ESTRUCTURA"); print("="*L)
    print(f"Unidad de analisis : {len(item):,} SKU (item_id, agregado sobre 10 tiendas)")
    print(f"Periodos completos : {meses[0]} -> {meses[-1]} ({len(meses)} meses)")
    print(f"Entrenamiento      : {col_tr[0]} -> {col_tr[-1]} ({n_tr} meses)")
    print(f"Evaluacion         : {col_te[0]} -> {col_te[-1]} ({H_TEST} meses)")
    print(f"Meses descartados  : 2011-01 (3 dias) y 2016-05 (22 dias)")
    print(f"\nCeldas totales     : {M.size:,}")
    print(f"Celdas vacias      : {int(np.isnan(M).sum()):,}")
    print(f"Celdas negativas   : {int((M < 0).sum()):,}")
    print(f"SKU duplicados     : {int(pd.Series(item).duplicated().sum())}")
    print(f"Precio no valido   : {int((precio <= 0).sum())} SKU")
    print(f"Unidades totales   : {int(M.sum()):,}  (de {int(V[meses_all].values.sum()):,} con meses parciales)")

    # ---------------------------------------------------------------- BLOQUE 2
    valor = M.sum(axis=1) * precio
    orden = np.argsort(-valor); acum = np.cumsum(valor[orden]) / valor.sum()
    print("\n" + "="*L); print("BLOQUE 2 — CONCENTRACION"); print("="*L)
    for u in (0.50, 0.80, 0.95):
        n = int((acum <= u).sum()) + 1
        print(f"  {u:.0%} del valor lo explican {n:>5,} SKU  ({n/len(item):>5.1%} del portafolio)")
    print(f"  SKU sin venta valorizada         : {int((valor == 0).sum()):,}")
    print(f"  Valor total del portafolio       : {valor.sum():,.0f}")

    # ---------------------------------------------------------------- BLOQUE 3
    T = M[:, :n_tr]
    adi = np.full(len(item), np.nan); cv = np.full(len(item), np.nan)
    cv2 = np.full(len(item), np.nan); nz_n = np.zeros(len(item), int)
    autoc = np.zeros(len(item)); estac = np.full(len(item), np.nan)
    for r in range(len(item)):
        h = T[r]; nz = h[h > 0]; nz_n[r] = len(nz)
        autoc[r] = pd.Series(h).autocorr(lag=1) if h.std() > 0 else 0.0
        if len(nz) < 2 or nz.mean() == 0: continue
        adi[r] = len(h) / len(nz)
        c = nz.std() / nz.mean(); cv[r] = c; cv2[r] = c**2
        if len(h) >= 36 and h[:36].reshape(3,12).mean() > 0:
            m36 = h[:36].reshape(3,12); estac[r] = (m36.mean(axis=0)/m36.mean()).std()

    e = pd.DataFrame({'SKU': item, 'DEPT': dept, 'CAT': cat, 'PRECIO': precio,
                      'VALOR': valor, 'MESES_VENTA': nz_n, 'ADI': adi, 'CV': cv,
                      'CV2': cv2, 'ESTAC': estac, 'AUTOCORR': np.nan_to_num(autoc)})

    print("\n" + "="*L); print(f"BLOQUE 3 — INTERMITENCIA Y VARIABILIDAD ({n_tr} meses de entrenamiento)"); print("="*L)
    print(f"  SKU sin historia suficiente (<2 meses con venta): {int(e['ADI'].isna().sum()):,}")
    print(f"\n  Meses con venta  mediana {e['MESES_VENTA'].median():>6.0f} de {n_tr} | "
          f"p25 {e['MESES_VENTA'].quantile(.25):.0f} | p75 {e['MESES_VENTA'].quantile(.75):.0f}")
    for nm, k in [('ADI','ADI'), ('CV','CV'), ('Intensidad estac','ESTAC')]:
        print(f"  {nm:<16} mediana {e[k].median():>6.2f} | p25 {e[k].quantile(.25):.2f} | p75 {e[k].quantile(.75):.2f}")

    # ---------------------------------------------------------------- BLOQUE 4
    def patron(a, c):
        if np.isnan(a) or np.isnan(c): return 'SIN DATOS'
        if a <  ADI_U and c <  CV2_U: return 'SUAVE'
        if a <  ADI_U and c >= CV2_U: return 'ERRATICO'
        if a >= ADI_U and c <  CV2_U: return 'INTERMITENTE'
        return 'CON PICOS'
    e['PATRON'] = [patron(a, c) for a, c in zip(e['ADI'], e['CV2'])]

    print("\n" + "="*L); print(f"BLOQUE 4 — PATRONES DE DEMANDA (Syntetos et al., 2005) | {len(e):,} SKU"); print("="*L)
    print(f"  {'PATRON':<14} {'SKU':>7} {'% SKU':>8} {'VALOR':>16} {'% VALOR':>9}")
    print("  " + "-"*58)
    tv = e['VALOR'].sum()
    for t in ['SUAVE','ERRATICO','INTERMITENTE','CON PICOS','SIN DATOS']:
        d = e[e['PATRON'] == t]
        if len(d) == 0: continue
        print(f"  {t:<14} {len(d):>7,} {len(d)/len(e):>7.1%} {d['VALOR'].sum():>16,.0f} {d['VALOR'].sum()/tv:>8.1%}")

    # ---------------------------------------------------------------- BLOQUE 5
    print("\n" + "="*L); print("BLOQUE 5 — CATEGORIA DEL PRODUCTO vs PATRON DE DEMANDA"); print("="*L)
    print(pd.crosstab(e['CAT'], e['PATRON']).to_string())
    reg = e['PATRON'].isin(['SUAVE','ERRATICO'])

    # ---------------------------------------------------------------- BLOQUE 6
    print("\n" + "="*L); print("BLOQUE 6 — EMBUDO DE ELEGIBILIDAD"); print("="*L)
    # La elegibilidad se define EXCLUSIVAMENTE con informacion disponible al cierre
    # del entrenamiento. El criterio de venta en el periodo de evaluacion queda
    # eliminado por introducir informacion del futuro en la definicion de la muestra.
    vivo_tr = T.sum(axis=1) > 0
    prim  = np.array([np.flatnonzero(T[r] > 0)[0] if (T[r] > 0).any() else 10**6
                      for r in range(len(item))])
    hist  = prim <= n_tr - 24
    pok   = precio > 0
    print("  NOTA: M5 no tiene indicador de liquidacion o remate. Ese criterio de")
    print("        exclusion del PAC no aplica y se declara como no aplicable.")
    print("  NOTA: la elegibilidad no consulta el periodo de evaluacion.\n")
    prev = len(e)
    for crit, mask in [('Sin venta en la ventana de entrenamiento', vivo_tr),
                       ('Historial menor a 24 meses', vivo_tr & hist),
                       ('Liquidacion o remate (no aplica en M5)', vivo_tr & hist),
                       ('Precio unitario no valido', vivo_tr & hist & pok)]:
        q = int(mask.sum()); print(f"  {crit:<42} descartan {prev-q:>5,} | quedan {q:>6,}"); prev = q
    eleg = vivo_tr & hist & pok
    _sin_test = int((eleg & reg.values & (M[:, n_tr:].sum(axis=1) == 0)).sum())
    print(f"  De la muestra, SKU sin venta en evaluacion: {_sin_test} "
          f"(se conservan; el conjunto de prueba solo se usa para evaluar)")
    mue  = eleg & reg.values
    print(f"\n  POBLACION DE ESTUDIO      : {int(eleg.sum()):,} SKU | {e.loc[eleg,'VALOR'].sum()/tv:.1%} del valor")
    print(f"  MUESTRA (suave + erratico): {int(mue.sum()):,} SKU | {e.loc[mue,'VALOR'].sum()/tv:.1%} del valor")
    print(f"  Descartados por patron    : {int((eleg & ~reg.values).sum()):,} SKU")
    u_tot = M.sum(); print(f"  Cobertura en unidades     : {M[mue].sum()/u_tot:.1%}")
    e['ELEGIBLE'] = eleg; e['MUESTRA'] = mue

    # ---------------------------------------------------------------- BLOQUE 7
    print("\n" + "="*L); print("BLOQUE 7 — HALLAZGO QUE SUSTENTA EL PROBLEMA"); print("="*L)
    alto = e['VALOR'] >= e['VALOR'].quantile(0.80)       # quintil superior de valor
    fuera = e[alto & eleg & ~reg]
    print(f"  SKU del 20 % superior de valor que NO son pronosticables por metodos")
    print(f"  regulares (patron intermitente o con picos): {len(fuera):,}")
    print(f"  Valor que representan: {fuera['VALOR'].sum():,.0f} ({fuera['VALOR'].sum()/tv:.1%} del portafolio)")

    # ---------------------------------------------------------------- BLOQUE 8
    idx = np.flatnonzero(mue)
    F = []
    for r in idx:
        h = T[r]; u12, p12 = h[-12:], h[-24:-12]
        F.append({'LAG_1': h[-1], 'LAG_12': h[-12], 'MEDIA_3': h[-3:].mean(),
                  'MEDIA_12': u12.mean(), 'FREC_12': int((u12 > 0).sum()),
                  'CV_12': u12.std()/u12.mean() if u12.mean() > 0 else 0,
                  'YOY': (u12.sum()/p12.sum()-1) if p12.sum() > 0 else 0,
                  'LOG_VALOR': np.log1p(u12.mean()*precio[r]),
                  'ESTAC': e['ESTAC'].iloc[r] if not np.isnan(e['ESTAC'].iloc[r]) else 0,
                  'AUTOCORR': e['AUTOCORR'].iloc[r]})
    F = pd.DataFrame(F).replace([np.inf,-np.inf], 0).fillna(0)
    print("\n" + "="*L); print(f"BLOQUE 8 — VARIABLES PREDICTORAS | n = {len(F):,} SKU de la muestra"); print("="*L)
    print("\n  Matriz de correlacion (Pearson):"); print(F.corr().round(2).to_string())
    from sklearn.linear_model import LinearRegression
    print(f"\n  {'VARIABLE':<12} {'FIV':>8}"); print("  " + "-"*22)
    Xs = (F - F.mean()) / F.std().replace(0, 1)
    for c in F.columns:
        r2 = LinearRegression().fit(Xs.drop(columns=[c]).values, Xs[c].values).score(
             Xs.drop(columns=[c]).values, Xs[c].values)
        print(f"  {c:<12} {(1/(1-r2) if r2 < 0.9999 else np.inf):>8.2f}")

    e.to_csv('m5_resultados_objetivo1.csv', index=False)
    np.save('m5_matriz.npy', M)
    pd.Series(meses).to_csv('m5_meses.csv', index=False, header=False)
    print("\nGuardado: m5_resultados_objetivo1.csv, m5_matriz.npy, m5_meses.csv")


# ===========================================================================
# CONJUNTO 2 — WALMART M5, OBJETIVOS ESPECIFICOS 2 y 3
# ===========================================================================
def correr_m5(_MERCADO=None):
    # =============================================================================
    # OBJETIVOS ESPECIFICOS 2 y 3 — M5 (Walmart)
    # Mismo protocolo que el conjunto de la empresa: K-means sobre 5 variables de
    # comportamiento, torneo RF/XGBoost por clúster con validación temporal, y
    # comparación contra promedio móvil y Holt-Winters.
    # Perfil móvil recalculado en cada corte: sin fuga de información.
    # =============================================================================
    import warnings, itertools, time; warnings.filterwarnings("ignore")
    import pandas as pd, numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, silhouette_score, adjusted_rand_score
    from xgboost import XGBRegressor
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from scipy.stats import wilcoxon

    SEED, H = 75, 12
    VARS = ['CV_12','ESTAC','YOY','LOG_VALOR','AUTOCORR']
    REJ_RF  = [{'n_estimators':400,'max_depth':d,'min_samples_leaf':l}
               for d,l in itertools.product([8,14,None],[1,3])]
    REJ_XGB = [{'n_estimators':n,'max_depth':d,'learning_rate':lr,'subsample':0.8,
                'colsample_bytree':0.8,'min_child_weight':w}
               for n,d,lr,w in itertools.product([400,800],[4,6],[0.03,0.08],[1,5])]

    E = pd.read_csv('m5_resultados_objetivo1.csv')
    M = np.load('m5_matriz.npy')
    meses = pd.read_csv('m5_meses.csv', header=None)[0].tolist()
    n_tr = len(meses) - H
    idx = np.flatnonzero(E['MUESTRA'].values)

    # ============================================================================
    # CRITERIOS DE NEGOCIO EN EL EMBUDO (ambos anteriores al conjunto de prueba)
    #  A) PRONOSTICABILIDAD  : CV de los periodos con venta del ultimo ciclo <= 1.0
    #     Umbral PREDECIBLE del simulador en produccion. Ventana: un ciclo estacional
    #     (12 meses en los cuatro conjuntos evaluados), por coherencia con LAG_12, MEDIA_12,
    #     FREC_12 y el horizonte de prueba. En produccion la ventana es de 24 meses
    #     porque alli responde a la politica de reposicion, no al ciclo de pronostico.
    #  B) ACTIVIDAD RECIENTE : venta en >= 6 periodos del ultimo ciclo Y en >= 4 de
    #     los ultimos 6 periodos (VENDE_6M y FREC_ULT6 del simulador).
    # Sustento: Brown (1959) recencia; Syntetos et al. (2005) patron; Boylan,
    # Syntetos y Karakostas (2008) revision de cortes; Teunter, Syntetos y Babai
    # (2011) obsolescencia; Bauer (2020) delimitacion por criterio de planificacion.
    # ============================================================================
    def _cv_ciclo(h):
        v = h[-12:]; nz = v[v > 0]
        return nz.std(ddof=1)/nz.mean() if len(nz) >= 2 and nz.mean() > 0 else 0.0
    _cv  = np.array([_cv_ciclo(M[r, :n_tr]) for r in idx])
    _f12 = np.array([(M[r, n_tr-12:n_tr] > 0).sum() for r in idx])
    _u6  = np.array([(M[r, n_tr-6:n_tr]    > 0).sum() for r in idx])
    _pred = _cv <= 1.0
    _act  = (_f12 >= 6) & (_u6 >= 4)
    print(f"EMBUDO DE NEGOCIO | no pronosticables (CV>1): {int((~_pred).sum()):,} | "
          f"sin actividad reciente: {int((_pred & ~_act).sum()):,} | "
          f"quedan {int((_pred & _act).sum()):,} de {len(idx):,}", flush=True)
    idx = idx[_pred & _act]

    precio = E['PRECIO'].values
    N = len(idx)
    t0 = time.time()
    print(f"Muestra: {N} SKU | train {n_tr} meses | test {H} meses", flush=True)

    # ----------------------------------------------------------------- perfil
    def f_est(y):
        if len(y) < 24 or np.std(y) == 0: return 0.0
        try:
            d = seasonal_decompose(pd.Series(y), model='additive', period=12, extrapolate_trend='freq')
            r_ = d.resid.dropna(); s_ = d.seasonal[r_.index]; v = np.var(r_+s_)
            return float(max(0.0, 1-np.var(r_)/v)) if v > 0 else 0.0
        except Exception: return 0.0

    def perfil(h, t, pu):
        v12 = h[t-12:t]; pv = h[t-24:t-12] if t >= 24 else np.array([0.0])
        return {'FREC_12': float((v12>0).sum()),
                'CV_12': v12.std()/v12.mean() if v12.mean()>0 else 0.0,
                'ESTAC': f_est(h[:t]),
                'YOY': (v12.sum()/pv.sum()-1) if pv.sum()>0 else 0.0,
                'LOG_VALOR': float(np.log1p(v12.mean()*pu)),
                'AUTOCORR': float(pd.Series(h[:t]).autocorr(lag=1)) if h[:t].std()>0 else 0.0}

    print("Precalculando perfiles móviles...", flush=True)
    MOV = {j: {t: perfil(M[r,:n_tr], t, precio[r]) for t in range(12, n_tr)}
           for j, r in enumerate(idx)}
    G0 = pd.DataFrame([perfil(M[r,:n_tr], n_tr, precio[r]) for r in idx]
                      ).replace([np.inf,-np.inf],0).fillna(0)
    G0['YOY'] = G0['YOY'].clip(G0['YOY'].quantile(.01), G0['YOY'].quantile(.99))
    Z = StandardScaler().fit_transform(G0[VARS])
    print(f"[{time.time()-t0:.0f}s] Listo.\n", flush=True)

    # ----------------------------------------------------------------- seleccion de K
    L = 96
    print("="*L); print("OE2 — SELECCION DEL NUMERO DE GRUPOS"); print("="*L)
    print(f"{'K':>3} {'SILUETA':>10} {'RAND AJUSTADO':>15} {'TAMANOS'}")
    print("-"*L)
    rng = np.random.default_rng(SEED); tabla = []
    for K in range(2, 9):
        lab = KMeans(n_clusters=K, random_state=SEED, n_init=20).fit(Z).labels_
        sil = float(silhouette_score(Z, lab))
        aris = []
        for _ in range(20):
            sub = rng.choice(N, int(0.8*N), replace=False)
            l2 = KMeans(n_clusters=K, random_state=SEED, n_init=20).fit(Z[sub]).labels_
            aris.append(adjusted_rand_score(lab[sub], l2))
        ari = float(np.mean(aris)); tam = [int((lab==c).sum()) for c in range(K)]
        tabla.append((K, sil, ari)); print(f"{K:>3} {sil:>10.3f} {ari:>15.3f} {tam}", flush=True)
    print("-"*L)
    # Regla de seleccion: mayor indice de Rand ajustado. Cuando dos o mas valores de K
    # difieren en menos de 0.01 -- por debajo de la resolucion de 20 remuestreos --
    # el empate se resuelve por el mayor indice de silueta. Ambos criterios son
    # anteriores al conjunto de prueba.
    # Piso de negocio: la segmentacion debe entregar al menos tres grupos con modelo.
    # Anclaje en produccion: el simulador de la empresa mantiene cuatro clusters, de
    # los cuales tres reciben modelo de pronostico y el cuarto se gobierna por cuota
    # comercial. Una particion con menos de tres grupos no es accionable para la
    # planificacion y por tanto no satisface el objetivo especifico 2. Criterio
    # anterior al conjunto de prueba.
    K_MIN = 3
    cand = [t for t in tabla if t[0] >= K_MIN]
    ari_max = max(t[2] for t in cand)
    empate = [t for t in cand if ari_max - t[2] < 0.01]
    K = max(empate, key=lambda x: x[1])[0]
    print(f"Piso de negocio: K >= {K_MIN} (tres clusters con modelo en produccion)")
    print(f"Mayor Rand ajustado entre los admisibles: {ari_max:.3f}. Empatados dentro de 0.01: "
          f"K = {[t[0] for t in empate]}")
    print(f"Desempate por silueta -> K = {K}")
    print("El conjunto de prueba no interviene en ninguno de los dos criterios.\n", flush=True)

    lab = KMeans(n_clusters=K, random_state=SEED, n_init=20).fit(Z).labels_
    val_sku = M[idx][:, :n_tr].sum(axis=1)*precio[idx]
    perf = G0[VARS].groupby(lab).mean()
    libres, nombre = list(perf.index), {}
    c = perf.loc[libres,'LOG_VALOR'].idxmax(); nombre[c]='Núcleo de alto valor'; libres.remove(c)
    if libres:
        c = perf.loc[libres,'YOY'].idxmax(); nombre[c]='Emergentes de crecimiento'; libres.remove(c)
    for i, c in enumerate(libres): nombre[c] = f'Cola volátil {i+1}' if len(libres)>1 else 'Cola volátil de bajo valor'
    ORDEN = sorted(range(K), key=lambda c: -val_sku[lab==c].sum())
    NUM = {c: i+1 for i, c in enumerate(ORDEN)}

    print("="*L); print("PERFIL DE LOS GRUPOS"); print("="*L)
    print(f"{'GRUPO':<30} {'SKU':>6} {'%SKU':>7} {'%VALOR':>8} "
          f"{'CV_12':>7} {'ESTAC':>7} {'YOY':>7} {'LOG_VAL':>8} {'AUTOC':>7}")
    print("-"*L)
    for c in ORDEN:
        m = lab==c; p = G0[VARS][m].mean()
        print(f"{NUM[c]}. {nombre[c]:<27} {int(m.sum()):>6} {m.mean():>7.1%} "
              f"{val_sku[m].sum()/val_sku.sum():>8.1%} {p['CV_12']:>7.2f} {p['ESTAC']:>7.2f} "
              f"{p['YOY']:>+7.2f} {p['LOG_VALOR']:>8.2f} {p['AUTOCORR']:>+7.2f}", flush=True)

    # ----------------------------------------------------------------- lineas base
    # ============================================================================
    # LINEA BASE 1 — PROMEDIO MOVIL (descomposicion clasica multiplicativa)
    # Funcion unica empleada en los cuatro conjuntos evaluados.
    #   1. Nivel = promedio del ultimo ciclo estacional.
    #   2. Indice estacional = promedio historico de cada posicion del ciclo
    #      dividido entre el promedio general de la serie.
    #   3. Crecimiento = promedio ponderado linealmente de los ratios interanuales
    #      entre ANIOS CIVILES CERRADOS (enero a diciembre) contenidos en la
    #      ventana de entrenamiento, con mayor peso en los mas recientes. Esta es
    #      la logica de negocio del sistema en produccion: el planificador compara
    #      anios cerrados, no ventanas moviles.
    #   4. El crecimiento se acota al rango [0.5, 1.5].
    #   5. El pronostico final tiene piso en cero.
    # Guardas: nivel no positivo devuelve ceros; posicion estacional con promedio
    # historico nulo recibe indice 1.0 en lugar de 0.0.
    # ============================================================================
    def pm(h, S, H, blq=None):
        nivel = h[-S:].mean()
        if nivel <= 0:
            return np.zeros(H)
        prom = h.mean() if h.mean() > 0 else 1.0
        ind = np.array([h[i::S].mean()/prom if h[i::S].mean() > 0 else 1.0 for i in range(S)])
        if blq:
            an = [h[b].sum() for b in blq]
        else:
            an = [h[i*S:(i+1)*S].sum() for i in range(len(h)//S)]
        rt = [an[i+1]/an[i] for i in range(len(an)-1) if an[i] > 0]
        g = float(np.clip(np.average(rt, weights=range(1, len(rt)+1)), 0.5, 1.5)) if rt else 1.0
        return np.array([max(0.0, nivel*ind[(len(h)+k) % S]*g) for k in range(H)])


    # Anios civiles cerrados (ENE-DIC) contenidos en la ventana de entrenamiento
    _an = {}
    for _i, _m in enumerate(meses[:n_tr]):
        _an.setdefault(str(_m)[:4], []).append(_i)
    CERRADOS = [a for a in sorted(_an) if len(_an[a]) == 12]
    BLQ = [_an[a] for a in CERRADOS]
    print(f"ANIOS CIVILES CERRADOS EN ENTRENAMIENTO: {CERRADOS} -> "
          f"{max(0,len(CERRADOS)-1)} ratio(s) interanual(es)", flush=True)

    def hw(h):
        try:
            if (h > 0).sum() < 24 or h.std() == 0: return np.repeat(h[-12:].mean(), H)
            f = ExponentialSmoothing(h, trend='add', seasonal='add', seasonal_periods=12,
                                     initialization_method='estimated').fit()
            return np.clip(f.forecast(H), 0, None)
        except Exception: return np.repeat(h[-12:].mean(), H)

    print(f"\n[{time.time()-t0:.0f}s] Corriendo líneas base...", flush=True)
    PM = np.array([pm(M[r,:n_tr], 12, H, BLQ) for r in idx])
    HW = np.array([hw(M[r,:n_tr]) for r in idx])
    print(f"[{time.time()-t0:.0f}s] Líneas base listas.", flush=True)

    # ----------------------------------------------------------------- torneo
    FIN_VAL = n_tr - H
    def fila(h, t, d):
        v12 = h[t-12:t]
        return [h[t-1], h[t-12], h[t-3:t].mean(), v12.mean(), (t%12)+1,
                d['FREC_12'], d['CV_12'], d['ESTAC'], d['YOY'], d['LOG_VALOR'], d['AUTOCORR']]

    NOMB_F = ['LAG_1', 'LAG_12', 'MEDIA_3', 'MEDIA_12', 'MES', 'FREC_12',
              'CV_12', 'ESTAC', 'YOY', 'LOG_VALOR', 'AUTOCORR']

    def entrenar(pos):
        Xtr,ytr,Xva,yva,pva = [],[],[],[],[]
        for j in pos:
            h = M[idx[j], :n_tr]
            for t in range(12, n_tr):
                f = fila(h, t, MOV[j][t])
                if t < FIN_VAL: Xtr.append(f); ytr.append(h[t])
                else: Xva.append(f); yva.append(h[t]); pva.append(precio[idx[j]])
        Xtr,ytr,Xva,yva,pva = map(np.array,(Xtr,ytr,Xva,yva,pva))
        wm = lambda yp: np.abs((yva-yp)*pva).sum()/max((yva*pva).sum(),1)
        b = {}
        for nm, rj, cl in [('RF',REJ_RF,RandomForestRegressor),('XGB',REJ_XGB,XGBRegressor)]:
            mj, hb = np.inf, None
            for hp in rj:
                w = wm(cl(random_state=SEED, n_jobs=-1, **hp).fit(Xtr,ytr).predict(Xva))
                if w < mj: mj, hb = w, hp
            b[nm] = (mj, hb)
        gan = 'RF' if b['RF'][0] <= b['XGB'][0] else 'XGB'
        cl = RandomForestRegressor if gan=='RF' else XGBRegressor
        mdl = cl(random_state=SEED, n_jobs=-1, **b[gan][1]).fit(
                  np.vstack([Xtr,Xva]), np.concatenate([ytr,yva]))
        return mdl, gan, b['RF'][0], b['XGB'][0]

    ML = np.zeros((N,H)); info = {}; IMP = {}
    print(f"\n[{time.time()-t0:.0f}s] Torneo por clúster...", flush=True)
    for c in ORDEN:
        pos = np.flatnonzero(lab==c)
        mdl, gan, wrf, wxg = entrenar(pos)
        info[c] = (gan, wrf, wxg, len(pos))
        IMP[c] = dict(zip(NOMB_F, mdl.feature_importances_))
        hist = [np.array(M[idx[j],:n_tr], dtype=float) for j in pos]
        for _ in range(H):
            Xb = [fila(hh, len(hh), perfil(hh, len(hh), precio[idx[j]]))
                  for hh, j in zip(hist, pos)]
            for i, v in enumerate(mdl.predict(np.array(Xb))):
                hist[i] = np.append(hist[i], max(0.0, float(v)))
        for i, j in enumerate(pos): ML[j] = hist[i][n_tr:n_tr+H]
        print(f"[{time.time()-t0:.0f}s]  grupo {NUM[c]}: {gan} | RF {wrf:.1%} vs XGB {wxg:.1%}", flush=True)

    print("\n" + "="*L); print("OE2 — TORNEO EN VALIDACION (WMAPE valorizado)"); print("="*L)
    print(f"{'GRUPO':<32} {'SKU':>6} {'RANDOM FOREST':>15} {'XGBOOST':>10} {'ASIGNADO':>10}")
    print("-"*L)
    for c in ORDEN:
        g, wrf, wxg, n = info[c]
        print(f"{NUM[c]}. {nombre[c]:<29} {n:>6} {wrf:>15.1%} {wxg:>10.1%} "
              f"{('Random Forest' if g=='RF' else 'XGBoost'):>10}")

    # Importancia de las variables en el modelo asignado a cada grupo.
    # Se reporta en los cuatro conjuntos con el mismo criterio, de modo que la
    # lectura de interpretabilidad no quede restringida a un solo portafolio.
    print("\nImportancia de variables por grupo (top 5):")
    for c in ORDEN:
        top = sorted(IMP[c].items(), key=lambda kv: -kv[1])[:5]
        print(f"  Grupo {NUM[c]}: " + " \u00b7 ".join(f"{k} {v:.3f}" for k, v in top))

    # ----------------------------------------------------------------- OE3
    real = M[np.ix_(idx, range(n_tr, n_tr+H))]; pr = precio[idx].reshape(-1,1)
    def met(mk, F):
        A,B = real[mk]*pr[mk], F[mk]*pr[mk]; U,Q = real[mk], F[mk]
        return (np.abs(A-B).sum()/max(A.sum(),1), B.sum()/max(A.sum(),1)-1,
                np.abs(U-Q).sum()/max(U.sum(),1), Q.sum()/max(U.sum(),1)-1,
                r2_score(U.ravel(), Q.ravel()))
    ea = lambda mk, F: (np.abs(real[mk]-F[mk])*pr[mk]).sum(axis=1)

    print("\n" + "="*L); print("OE3 — DESEMPENO EN EL CONJUNTO DE PRUEBA"); print("="*L)
    print(f"{'GRUPO':<26} {'%VAL':>6} {'METODO':<16} {'WMAPE$':>8} {'BIAS$':>8} "
          f"{'WMAPEu':>8} {'BIASu':>8} {'R2':>6} {'p vs PM':>9} {'p vs HW':>9}")
    print("-"*L)
    filas = []
    for c in list(ORDEN) + ['TOTAL']:
        mk = np.ones(N,bool) if c=='TOTAL' else (lab==c)
        et = 'AGREGADO' if c=='TOTAL' else f"{NUM[c]}. {nombre[c]}"
        pv = val_sku[mk].sum()/val_sku.sum()
        for nm, F in [('Promedio móvil',PM), ('Holt-Winters',HW), ('Arquitectura',ML)]:
            wv,bv,wu,bu,r2 = met(mk,F)
            if nm=='Arquitectura':
                p1 = wilcoxon(ea(mk,PM), ea(mk,ML))[1]; p2 = wilcoxon(ea(mk,HW), ea(mk,ML))[1]
                s1, s2 = f"{p1:.4f}", f"{p2:.4f}"
            else: s1 = s2 = '—'
            print(f"{et if nm=='Promedio móvil' else '':<26} "
                  f"{(f'{pv:.1%}' if nm=='Promedio móvil' else ''):>6} {nm:<16} "
                  f"{wv:>8.1%} {bv:>+8.1%} {wu:>8.1%} {bu:>+8.1%} {r2:>6.2f} {s1:>9} {s2:>9}")
            filas.append(dict(grupo=et, peso=pv, metodo=nm, wmape_val=wv, bias_val=bv,
                              wmape_u=wu, bias_u=bu, r2=r2, p_pm=s1, p_hw=s2))
        print("-"*L)

    d_pm = (np.abs(real-PM)*pr).sum() - (np.abs(real-ML)*pr).sum()
    d_hw = (np.abs(real-HW)*pr).sum() - (np.abs(real-ML)*pr).sum()
    print(f"\nReducción acumulada del error valorizado frente al promedio móvil: {d_pm:>+15,.0f}")
    print(f"Reducción acumulada del error valorizado frente a Holt-Winters   : {d_hw:>+15,.0f}")

    pd.DataFrame(filas).to_csv('v6_m5_oe3.csv', index=False)
    pd.DataFrame({'sku': E['SKU'].values[idx], 'cluster': [NUM[c] for c in lab],
                  'nombre': [nombre[c] for c in lab]}).to_csv('v6_m5_clusters.csv', index=False)
    np.save('v6_m5_ml.npy', ML); np.save('v6_m5_pm.npy', PM); np.save('v6_m5_hw.npy', HW)
    print(f"\n[{time.time()-t0:.0f}s] Guardado: m5_resultados_oe3.csv, m5_clusters.csv, predicciones .npy")


# ===========================================================================
# CONJUNTOS 3 y 4 — ROHLIK GROUP, OBJETIVO ESPECIFICO 1
# ===========================================================================
def rohlik_oe1(_MERCADO=None):
    # =============================================================================
    # OBJETIVO ESPECIFICO 1 — ROHLIK (e-grocery Europa Central)
    # Mismo protocolo declarado en el PAC para el conjunto de la empresa y para M5.
    # Unidad de analisis: product_unique_id, agregando los almacenes del mercado.
    # Mercados tratados como fuentes independientes: Chequia y Hungria.
    # =============================================================================
    import warnings, sys; warnings.filterwarnings("ignore")
    import pandas as pd, numpy as np

    PAIS = _MERCADO
    U = ''   # las rutas se resuelven con ruta(); ver CARPETAS al inicio
    ARCH = {'cz': ('rohlik_cz_mensual.csv', 'rohlik_cz_precios.csv', 'Chequia', 'CZK'),
            'hu': ('rohlik_hu_mensual.csv', 'rohlik_hu_precios.csv', 'Hungria', 'HUF')}
    fv, fp, NOM, MON = ARCH[PAIS]

    ADI_U, CV2_U = 1.32, 0.49          # umbrales Syntetos et al. (2005)
    H_TEST = 12                        # meses de evaluacion, igual que el privado y M5

    V = pd.read_csv(ruta(U + fv))
    P = pd.read_csv(ruta(U + fp))

    meses_all = [c for c in V.columns if c[:2] == '20']
    # --- deteccion de meses incompletos (misma regla que M5: 2011-01 y 2016-05) ---
    tot = V[meses_all].sum()
    med = tot.median()
    INCOMPLETOS = [m for m in meses_all if tot[m] < 0.35*med]
    meses = [m for m in meses_all if m not in INCOMPLETOS]
    n_tr = len(meses) - H_TEST
    col_tr, col_te = meses[:n_tr], meses[n_tr:]

    M = V[meses].values.astype(float)
    item = V['product_unique_id'].values
    alm  = V['n_almacenes'].values
    precio = P.set_index('product_unique_id').loc[item, 'precio_medio'].values

    L = 78
    print("="*L); print(f"BLOQUE 0 — ESTRUCTURA | ROHLIK {NOM.upper()}"); print("="*L)
    print(f"Unidad de analisis : {len(item):,} productos (product_unique_id, agregado sobre almacenes)")
    print(f"Almacenes por prod : mediana {np.median(alm):.0f} | maximo {alm.max()}")
    print(f"Periodos crudos    : {meses_all[0]} -> {meses_all[-1]} ({len(meses_all)} meses)")
    print(f"Meses descartados  : {INCOMPLETOS} por cobertura parcial "
          f"({', '.join(f'{tot[m]/med:.1%} de un mes tipico' for m in INCOMPLETOS)})")
    print(f"Periodos completos : {meses[0]} -> {meses[-1]} ({len(meses)} meses)")
    print(f"Entrenamiento      : {col_tr[0]} -> {col_tr[-1]} ({n_tr} meses)")
    print(f"Evaluacion         : {col_te[0]} -> {col_te[-1]} ({H_TEST} meses)")
    anios_tr = sorted({m[:4] for m in col_tr})
    cerrados = [a for a in anios_tr if sum(1 for m in col_tr if m[:4] == a) == 12]
    print(f"Anios civiles completos dentro del entrenamiento: {cerrados if cerrados else 'ninguno'}")
    print(f"  -> ratios interanuales disponibles para el crecimiento del promedio movil: "
          f"{max(0, len(cerrados)-1)}")
    print(f"\nCeldas totales     : {M.size:,}")
    print(f"Celdas vacias      : {int(np.isnan(M).sum()):,}")
    print(f"Celdas negativas   : {int((M < 0).sum()):,}")
    print(f"SKU duplicados     : {int(pd.Series(item).duplicated().sum())}")
    print(f"Precio no valido   : {int((precio <= 0).sum())} productos")
    print(f"Unidades totales   : {M.sum():,.0f}  (de {V[meses_all].values.sum():,.0f} con meses parciales)")
    print(f"Moneda del precio  : {MON}")

    # ---------------------------------------------------------------- BLOQUE 2
    valor = M.sum(axis=1) * precio
    orden = np.argsort(-valor); acum = np.cumsum(valor[orden]) / valor.sum()
    print("\n" + "="*L); print("BLOQUE 2 — CONCENTRACION"); print("="*L)
    for u in (0.50, 0.80, 0.95):
        n = int((acum <= u).sum()) + 1
        print(f"  {u:.0%} del valor lo explican {n:>5,} productos  ({n/len(item):>5.1%} del portafolio)")
    print(f"  Productos sin venta valorizada   : {int((valor == 0).sum()):,}")
    print(f"  Valor total del portafolio       : {valor.sum():,.0f} {MON}")

    # ---------------------------------------------------------------- BLOQUE 3
    T = M[:, :n_tr]
    adi = np.full(len(item), np.nan); cv = np.full(len(item), np.nan)
    cv2 = np.full(len(item), np.nan); nz_n = np.zeros(len(item), int)
    autoc = np.zeros(len(item)); estac = np.full(len(item), np.nan)
    for r in range(len(item)):
        h = T[r]; nz = h[h > 0]; nz_n[r] = len(nz)
        autoc[r] = pd.Series(h).autocorr(lag=1) if h.std() > 0 else 0.0
        if len(nz) < 2 or nz.mean() == 0: continue
        adi[r] = len(h) / len(nz)
        c = nz.std() / nz.mean(); cv[r] = c; cv2[r] = c**2
        nb = len(h)//12
        if nb >= 2 and h[:nb*12].reshape(nb,12).mean() > 0:
            mb = h[:nb*12].reshape(nb,12); estac[r] = (mb.mean(axis=0)/mb.mean()).std()

    e = pd.DataFrame({'SKU': item, 'ALMACENES': alm, 'PRECIO': precio,
                      'VALOR': valor, 'MESES_VENTA': nz_n, 'ADI': adi, 'CV': cv,
                      'CV2': cv2, 'ESTAC': estac, 'AUTOCORR': np.nan_to_num(autoc)})

    print("\n" + "="*L); print(f"BLOQUE 3 — INTERMITENCIA Y VARIABILIDAD ({n_tr} meses de entrenamiento)"); print("="*L)
    print(f"  Productos sin historia suficiente (<2 meses con venta): {int(e['ADI'].isna().sum()):,}")
    print(f"\n  Meses con venta  mediana {e['MESES_VENTA'].median():>6.0f} de {n_tr} | "
          f"p25 {e['MESES_VENTA'].quantile(.25):.0f} | p75 {e['MESES_VENTA'].quantile(.75):.0f}")
    for nm, k in [('ADI','ADI'), ('CV','CV'), ('Intensidad estac','ESTAC')]:
        print(f"  {nm:<16} mediana {e[k].median():>6.2f} | p25 {e[k].quantile(.25):.2f} | p75 {e[k].quantile(.75):.2f}")

    # ---------------------------------------------------------------- BLOQUE 4
    def patron(a, c):
        if np.isnan(a) or np.isnan(c): return 'SIN DATOS'
        if a <  ADI_U and c <  CV2_U: return 'SUAVE'
        if a <  ADI_U and c >= CV2_U: return 'ERRATICO'
        if a >= ADI_U and c <  CV2_U: return 'INTERMITENTE'
        return 'CON PICOS'
    e['PATRON'] = [patron(a, c) for a, c in zip(e['ADI'], e['CV2'])]

    print("\n" + "="*L); print(f"BLOQUE 4 — PATRONES DE DEMANDA (Syntetos et al., 2005) | {len(e):,} productos"); print("="*L)
    print(f"  {'PATRON':<14} {'SKU':>7} {'% SKU':>8} {'VALOR':>16} {'% VALOR':>9}")
    print("  " + "-"*58)
    tv = e['VALOR'].sum()
    for t in ['SUAVE','ERRATICO','INTERMITENTE','CON PICOS','SIN DATOS']:
        d = e[e['PATRON'] == t]
        if len(d) == 0: continue
        print(f"  {t:<14} {len(d):>7,} {len(d)/len(e):>7.1%} {d['VALOR'].sum():>16,.0f} {d['VALOR'].sum()/tv:>8.1%}")
    reg = e['PATRON'].isin(['SUAVE','ERRATICO'])

    # ---------------------------------------------------------------- BLOQUE 6
    print("\n" + "="*L); print("BLOQUE 6 — EMBUDO DE ELEGIBILIDAD"); print("="*L)
    # La elegibilidad se define EXCLUSIVAMENTE con informacion disponible al cierre
    # de la ventana de entrenamiento. El criterio "registrar venta en el periodo de
    # evaluacion" queda eliminado: consultar el TEST para decidir quien entra a la
    # muestra introduce informacion del futuro en la definicion del objeto evaluado.
    # La inactividad se detecta con los criterios de actividad reciente calculados
    # sobre TRAIN (venta en >= 6 de los ultimos 12 periodos y en >= 4 de los ultimos 6).
    prim  = np.array([np.flatnonzero(T[r] > 0)[0] if (T[r] > 0).any() else 10**6
                      for r in range(len(item))])
    hist  = prim <= n_tr - 24
    pok   = precio > 0
    vivo_tr = T.sum(axis=1) > 0
    print("  NOTA: Rohlik no publica indicador de liquidacion o remate. Ese criterio de")
    print("        exclusion del PAC no aplica y se declara como no aplicable.")
    print("  NOTA: la elegibilidad no consulta el periodo de evaluacion.\n")
    prev = len(e)
    for crit, mask in [('Sin venta en la ventana de entrenamiento', vivo_tr),
                       ('Historial menor a 24 meses', vivo_tr & hist),
                       ('Liquidacion o remate (no aplica en Rohlik)', vivo_tr & hist),
                       ('Precio unitario no valido', vivo_tr & hist & pok)]:
        q = int(mask.sum()); print(f"  {crit:<42} descartan {prev-q:>5,} | quedan {q:>6,}"); prev = q
    eleg = vivo_tr & hist & pok
    _sin_test = int((eleg & reg.values & (M[:, n_tr:].sum(axis=1) == 0)).sum())
    print(f"  De la muestra, productos sin venta en evaluacion: {_sin_test} "
          f"(se conservan; el TEST solo se usa para evaluar)")
    mue  = eleg & reg.values
    print(f"\n  POBLACION DE ESTUDIO      : {int(eleg.sum()):,} productos | {e.loc[eleg,'VALOR'].sum()/tv:.1%} del valor")
    print(f"  MUESTRA (suave + erratico): {int(mue.sum()):,} productos | {e.loc[mue,'VALOR'].sum()/tv:.1%} del valor")
    print(f"  Descartados por patron    : {int((eleg & ~reg.values).sum()):,} productos")
    u_tot = M.sum(); print(f"  Cobertura en unidades     : {M[mue].sum()/u_tot:.1%}")
    e['ELEGIBLE'] = eleg; e['MUESTRA'] = mue

    # ---------------------------------------------------------------- BLOQUE 7
    print("\n" + "="*L); print("BLOQUE 7 — HALLAZGO QUE SUSTENTA EL PROBLEMA"); print("="*L)
    alto = e['VALOR'] >= e['VALOR'].quantile(0.80)
    fuera = e[alto & eleg & ~reg]
    print(f"  Productos del 20 % superior de valor que NO son pronosticables por metodos")
    print(f"  regulares (patron intermitente o con picos): {len(fuera):,}")
    print(f"  Valor que representan: {fuera['VALOR'].sum():,.0f} ({fuera['VALOR'].sum()/tv:.1%} del portafolio)")

    e.to_csv(f'roh_{PAIS}_objetivo1.csv', index=False)
    np.save(f'roh_{PAIS}_matriz.npy', M)
    pd.Series(meses).to_csv(f'roh_{PAIS}_meses.csv', index=False, header=False)
    print(f"\nGuardado: roh_{PAIS}_objetivo1.csv, roh_{PAIS}_matriz.npy, roh_{PAIS}_meses.csv")


# ===========================================================================
# CONJUNTOS 3 y 4 — ROHLIK GROUP, OBJETIVOS ESPECIFICOS 2 y 3
# ===========================================================================
def rohlik_oe2_oe3(_MERCADO=None):
    # =============================================================================
    # OBJETIVOS ESPECIFICOS 2 y 3 — ROHLIK (Chequia / Hungria)
    # Mismo protocolo que el conjunto de la empresa y que M5: K-means sobre 5
    # variables de comportamiento, torneo RF/XGBoost por cluster con validacion
    # temporal, y comparacion contra promedio movil y Holt-Winters.
    # Perfil movil recalculado en cada corte: sin fuga de informacion.
    # =============================================================================
    import warnings, itertools, time, sys; warnings.filterwarnings("ignore")
    import pandas as pd, numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, silhouette_score, adjusted_rand_score
    from xgboost import XGBRegressor
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from scipy.stats import wilcoxon

    PAIS = _MERCADO
    NOM = {'cz': 'Chequia', 'hu': 'Hungria'}[PAIS]
    SEED, H = 75, 12
    VARS = ['CV_12','ESTAC','YOY','LOG_VALOR','AUTOCORR']
    REJ_RF  = [{'n_estimators':400,'max_depth':d,'min_samples_leaf':l}
               for d,l in itertools.product([8,14,None],[1,3])]
    REJ_XGB = [{'n_estimators':n,'max_depth':d,'learning_rate':lr,'subsample':0.8,
                'colsample_bytree':0.8,'min_child_weight':w}
               for n,d,lr,w in itertools.product([400,800],[4,6],[0.03,0.08],[1,5])]

    E = pd.read_csv(f'roh_{PAIS}_objetivo1.csv')
    M = np.load(f'roh_{PAIS}_matriz.npy')
    meses = pd.read_csv(f'roh_{PAIS}_meses.csv', header=None)[0].astype(str).tolist()
    n_tr = len(meses) - H
    idx = np.flatnonzero(E['MUESTRA'].values)

    # ============================================================================
    # CRITERIOS DE NEGOCIO EN EL EMBUDO (ambos anteriores al conjunto de prueba)
    #  A) PRONOSTICABILIDAD  : CV de los periodos con venta del ultimo ciclo <= 1.0
    #  B) ACTIVIDAD RECIENTE : venta en >= 6 periodos del ultimo ciclo Y en >= 4 de
    #     los ultimos 6 periodos (VENDE_6M y FREC_ULT6 del simulador).
    # ============================================================================
    def _cv_ciclo(h):
        v = h[-12:]; nz = v[v > 0]
        return nz.std(ddof=1)/nz.mean() if len(nz) >= 2 and nz.mean() > 0 else 0.0
    _cv  = np.array([_cv_ciclo(M[r, :n_tr]) for r in idx])
    _f12 = np.array([(M[r, n_tr-12:n_tr] > 0).sum() for r in idx])
    _u6  = np.array([(M[r, n_tr-6:n_tr]    > 0).sum() for r in idx])
    _pred = _cv <= 1.0
    _act  = (_f12 >= 6) & (_u6 >= 4)
    print(f"ROHLIK {NOM.upper()}", flush=True)
    print(f"EMBUDO DE NEGOCIO | no pronosticables (CV>1): {int((~_pred).sum()):,} | "
          f"sin actividad reciente: {int((_pred & ~_act).sum()):,} | "
          f"quedan {int((_pred & _act).sum()):,} de {len(idx):,}", flush=True)
    idx = idx[_pred & _act]

    precio = E['PRECIO'].values
    N = len(idx)
    t0 = time.time()
    print(f"Muestra: {N} SKU | train {n_tr} meses ({meses[0]} -> {meses[n_tr-1]}) | "
          f"test {H} meses ({meses[n_tr]} -> {meses[-1]})", flush=True)

    # ----------------------------------------------------------------- perfil
    def f_est(y):
        if len(y) < 24 or np.std(y) == 0: return 0.0
        try:
            d = seasonal_decompose(pd.Series(y), model='additive', period=12, extrapolate_trend='freq')
            r_ = d.resid.dropna(); s_ = d.seasonal[r_.index]; v = np.var(r_+s_)
            return float(max(0.0, 1-np.var(r_)/v)) if v > 0 else 0.0
        except Exception: return 0.0

    def perfil(h, t, pu):
        v12 = h[t-12:t]; pv = h[t-24:t-12] if t >= 24 else np.array([0.0])
        return {'FREC_12': float((v12>0).sum()),
                'CV_12': v12.std()/v12.mean() if v12.mean()>0 else 0.0,
                'ESTAC': f_est(h[:t]),
                'YOY': (v12.sum()/pv.sum()-1) if pv.sum()>0 else 0.0,
                'LOG_VALOR': float(np.log1p(v12.mean()*pu)),
                'AUTOCORR': float(pd.Series(h[:t]).autocorr(lag=1)) if h[:t].std()>0 else 0.0}

    print("Precalculando perfiles moviles...", flush=True)
    MOV = {j: {t: perfil(M[r,:n_tr], t, precio[r]) for t in range(12, n_tr)}
           for j, r in enumerate(idx)}
    G0 = pd.DataFrame([perfil(M[r,:n_tr], n_tr, precio[r]) for r in idx]
                      ).replace([np.inf,-np.inf],0).fillna(0)
    G0['YOY'] = G0['YOY'].clip(G0['YOY'].quantile(.01), G0['YOY'].quantile(.99))
    Z = StandardScaler().fit_transform(G0[VARS])
    print(f"[{time.time()-t0:.0f}s] Listo.\n", flush=True)

    # ----------------------------------------------------------------- seleccion de K
    L = 96
    print("="*L); print("OE2 — SELECCION DEL NUMERO DE GRUPOS"); print("="*L)
    print(f"{'K':>3} {'SILUETA':>10} {'RAND AJUSTADO':>15} {'TAMANOS'}")
    print("-"*L)
    rng = np.random.default_rng(SEED); tabla = []
    for K in range(2, 9):
        lab = KMeans(n_clusters=K, random_state=SEED, n_init=20).fit(Z).labels_
        sil = float(silhouette_score(Z, lab))
        aris = []
        for _ in range(20):
            sub = rng.choice(N, int(0.8*N), replace=False)
            l2 = KMeans(n_clusters=K, random_state=SEED, n_init=20).fit(Z[sub]).labels_
            aris.append(adjusted_rand_score(lab[sub], l2))
        ari = float(np.mean(aris)); tam = [int((lab==c).sum()) for c in range(K)]
        tabla.append((K, sil, ari)); print(f"{K:>3} {sil:>10.3f} {ari:>15.3f} {tam}", flush=True)
    print("-"*L)
    # Piso de negocio: la segmentacion debe entregar al menos tres grupos con modelo.
    # Anclaje en produccion: el simulador mantiene cuatro clusters, de los cuales tres
    # reciben modelo de pronostico y el cuarto se gobierna por cuota comercial. Una
    # particion con menos de tres grupos no es accionable para la planificacion y por
    # tanto no satisface el objetivo especifico 2. Criterio anterior al conjunto de prueba.
    K_MIN = 3
    cand = [t for t in tabla if t[0] >= K_MIN]
    ari_max = max(t[2] for t in cand)
    empate = [t for t in cand if ari_max - t[2] < 0.01]
    K = max(empate, key=lambda x: x[1])[0]
    print(f"Piso de negocio: K >= {K_MIN} (tres clusters con modelo en produccion)")
    print(f"Mayor Rand ajustado entre los admisibles: {ari_max:.3f}. Empatados dentro de 0.01: "
          f"K = {[t[0] for t in empate]}")
    print(f"Desempate por silueta -> K = {K}")
    print("El conjunto de prueba no interviene en ninguno de los dos criterios.\n", flush=True)

    lab = KMeans(n_clusters=K, random_state=SEED, n_init=20).fit(Z).labels_
    val_sku = M[idx][:, :n_tr].sum(axis=1)*precio[idx]
    perf = G0[VARS].groupby(lab).mean()
    libres, nombre = list(perf.index), {}
    c = perf.loc[libres,'LOG_VALOR'].idxmax(); nombre[c]='Núcleo de alto valor'; libres.remove(c)
    if libres:
        c = perf.loc[libres,'YOY'].idxmax(); nombre[c]='Emergentes de crecimiento'; libres.remove(c)
    for i, c in enumerate(libres): nombre[c] = f'Cola volátil {i+1}' if len(libres)>1 else 'Cola volátil de bajo valor'
    ORDEN = sorted(range(K), key=lambda c: -val_sku[lab==c].sum())
    NUM = {c: i+1 for i, c in enumerate(ORDEN)}

    print("="*L); print("PERFIL DE LOS GRUPOS"); print("="*L)
    print(f"{'GRUPO':<30} {'SKU':>6} {'%SKU':>7} {'%VALOR':>8} "
          f"{'CV_12':>7} {'ESTAC':>7} {'YOY':>7} {'LOG_VAL':>8} {'AUTOC':>7}")
    print("-"*L)
    for c in ORDEN:
        m = lab==c; p = G0[VARS][m].mean()
        print(f"{NUM[c]}. {nombre[c]:<27} {int(m.sum()):>6} {m.mean():>7.1%} "
              f"{val_sku[m].sum()/val_sku.sum():>8.1%} {p['CV_12']:>7.2f} {p['ESTAC']:>7.2f} "
              f"{p['YOY']:>+7.2f} {p['LOG_VALOR']:>8.2f} {p['AUTOCORR']:>+7.2f}", flush=True)

    # ----------------------------------------------------------------- lineas base
    # Anios civiles cerrados dentro del entrenamiento (logica de negocio del privado)
    anios = {}
    for i, m in enumerate(meses[:n_tr]):
        anios.setdefault(m[:4], []).append(i)
    CERR = [a for a in sorted(anios) if len(anios[a]) == 12]
    BLQ = [anios[a] for a in CERR]
    print(f"\nANIOS CIVILES CERRADOS EN ENTRENAMIENTO: {CERR} -> "
          f"{max(0,len(CERR)-1)} ratio(s) interanual(es)", flush=True)

    def pm(h, S, H, blq=None):
        """Descomposicion clasica multiplicativa. Crecimiento sobre anios civiles
        cerrados (blq) cuando se dispone de ellos; en su defecto, bloques de S
        desde el inicio de la historia. Funcion unica para los tres conjuntos."""
        nivel = h[-S:].mean()
        if nivel <= 0: return np.zeros(H)
        prom = h.mean() if h.mean() > 0 else 1.0
        ind = np.array([h[i::S].mean()/prom if h[i::S].mean() > 0 else 1.0 for i in range(S)])
        if blq: an = [h[b].sum() for b in blq]
        else:   an = [h[i*S:(i+1)*S].sum() for i in range(len(h)//S)]
        rt = [an[i+1]/an[i] for i in range(len(an)-1) if an[i] > 0]
        g = float(np.clip(np.average(rt, weights=range(1, len(rt)+1)), 0.5, 1.5)) if rt else 1.0
        return np.array([max(0.0, nivel*ind[(len(h)+k) % S]*g) for k in range(H)])

    def hw(h):
        try:
            if (h > 0).sum() < 24 or h.std() == 0: return np.repeat(h[-12:].mean(), H)
            f = ExponentialSmoothing(h, trend='add', seasonal='add', seasonal_periods=12,
                                     initialization_method='estimated').fit()
            return np.clip(f.forecast(H), 0, None)
        except Exception: return np.repeat(h[-12:].mean(), H)

    print(f"\n[{time.time()-t0:.0f}s] Corriendo lineas base...", flush=True)
    PM  = np.array([pm(M[r,:n_tr], 12, H, BLQ) for r in idx])     # anios civiles
    PMb = np.array([pm(M[r,:n_tr], 12, H, None) for r in idx])    # bloques desde el inicio
    HW  = np.array([hw(M[r,:n_tr]) for r in idx])
    print(f"[{time.time()-t0:.0f}s] Lineas base listas.", flush=True)

    # ----------------------------------------------------------------- torneo
    FIN_VAL = n_tr - H
    def fila(h, t, d):
        v12 = h[t-12:t]
        return [h[t-1], h[t-12], h[t-3:t].mean(), v12.mean(), (t%12)+1,
                d['FREC_12'], d['CV_12'], d['ESTAC'], d['YOY'], d['LOG_VALOR'], d['AUTOCORR']]

    NOMB_F = ['LAG_1', 'LAG_12', 'MEDIA_3', 'MEDIA_12', 'MES', 'FREC_12',
              'CV_12', 'ESTAC', 'YOY', 'LOG_VALOR', 'AUTOCORR']

    def entrenar(pos):
        Xtr,ytr,Xva,yva,pva = [],[],[],[],[]
        for j in pos:
            h = M[idx[j], :n_tr]
            for t in range(12, n_tr):
                f = fila(h, t, MOV[j][t])
                if t < FIN_VAL: Xtr.append(f); ytr.append(h[t])
                else: Xva.append(f); yva.append(h[t]); pva.append(precio[idx[j]])
        Xtr,ytr,Xva,yva,pva = map(np.array,(Xtr,ytr,Xva,yva,pva))
        wm = lambda yp: np.abs((yva-yp)*pva).sum()/max((yva*pva).sum(),1)
        b = {}
        for nm, rj, cl in [('RF',REJ_RF,RandomForestRegressor),('XGB',REJ_XGB,XGBRegressor)]:
            mj, hb = np.inf, None
            for hp in rj:
                w = wm(cl(random_state=SEED, n_jobs=-1, **hp).fit(Xtr,ytr).predict(Xva))
                if w < mj: mj, hb = w, hp
            b[nm] = (mj, hb)
        gan = 'RF' if b['RF'][0] <= b['XGB'][0] else 'XGB'
        cl = RandomForestRegressor if gan=='RF' else XGBRegressor
        mdl = cl(random_state=SEED, n_jobs=-1, **b[gan][1]).fit(
                  np.vstack([Xtr,Xva]), np.concatenate([ytr,yva]))
        return mdl, gan, b['RF'][0], b['XGB'][0]

    ML = np.zeros((N,H)); info = {}; IMP = {}
    print(f"\n[{time.time()-t0:.0f}s] Torneo por cluster...", flush=True)
    for c in ORDEN:
        pos = np.flatnonzero(lab==c)
        mdl, gan, wrf, wxg = entrenar(pos)
        info[c] = (gan, wrf, wxg, len(pos))
        IMP[c] = dict(zip(NOMB_F, mdl.feature_importances_))
        hist = [np.array(M[idx[j],:n_tr], dtype=float) for j in pos]
        for _ in range(H):
            Xb = [fila(hh, len(hh), perfil(hh, len(hh), precio[idx[j]]))
                  for hh, j in zip(hist, pos)]
            for i, v in enumerate(mdl.predict(np.array(Xb))):
                hist[i] = np.append(hist[i], max(0.0, float(v)))
        for i, j in enumerate(pos): ML[j] = hist[i][n_tr:n_tr+H]
        print(f"[{time.time()-t0:.0f}s]  grupo {NUM[c]}: {gan} | RF {wrf:.1%} vs XGB {wxg:.1%}", flush=True)

    print("\n" + "="*L); print("OE2 — TORNEO EN VALIDACION (WMAPE valorizado)"); print("="*L)
    print(f"{'GRUPO':<32} {'SKU':>6} {'RANDOM FOREST':>15} {'XGBOOST':>10} {'ASIGNADO':>10}")
    print("-"*L)
    for c in ORDEN:
        g, wrf, wxg, n = info[c]
        print(f"{NUM[c]}. {nombre[c]:<29} {n:>6} {wrf:>15.1%} {wxg:>10.1%} "
              f"{('Random Forest' if g=='RF' else 'XGBoost'):>10}")

    # Importancia de las variables en el modelo asignado a cada grupo.
    # Se reporta en los cuatro conjuntos con el mismo criterio, de modo que la
    # lectura de interpretabilidad no quede restringida a un solo portafolio.
    print("\nImportancia de variables por grupo (top 5):")
    for c in ORDEN:
        top = sorted(IMP[c].items(), key=lambda kv: -kv[1])[:5]
        print(f"  Grupo {NUM[c]}: " + " \u00b7 ".join(f"{k} {v:.3f}" for k, v in top))

    # ----------------------------------------------------------------- OE3
    real = M[np.ix_(idx, range(n_tr, n_tr+H))]; pr = precio[idx].reshape(-1,1)
    def met(mk, F):
        A,B = real[mk]*pr[mk], F[mk]*pr[mk]; U,Q = real[mk], F[mk]
        return (np.abs(A-B).sum()/max(A.sum(),1), B.sum()/max(A.sum(),1)-1,
                np.abs(U-Q).sum()/max(U.sum(),1), Q.sum()/max(U.sum(),1)-1,
                r2_score(U.ravel(), Q.ravel()))
    ea = lambda mk, F: (np.abs(real[mk]-F[mk])*pr[mk]).sum(axis=1)

    print("\n" + "="*L); print("OE3 — DESEMPENO EN EL CONJUNTO DE PRUEBA"); print("="*L)
    print(f"{'GRUPO':<26} {'%VAL':>6} {'METODO':<16} {'WMAPE$':>8} {'BIAS$':>8} "
          f"{'WMAPEu':>8} {'BIASu':>8} {'R2':>6} {'p vs PM':>9} {'p vs HW':>9}")
    print("-"*L)
    filas = []
    for c in list(ORDEN) + ['TOTAL']:
        mk = np.ones(N,bool) if c=='TOTAL' else (lab==c)
        et = 'AGREGADO' if c=='TOTAL' else f"{NUM[c]}. {nombre[c]}"
        pv = val_sku[mk].sum()/val_sku.sum()
        for nm, F in [('Promedio móvil',PM), ('Holt-Winters',HW), ('Arquitectura',ML)]:
            wv,bv,wu,bu,r2 = met(mk,F)
            if nm=='Arquitectura':
                p1 = wilcoxon(ea(mk,PM), ea(mk,ML))[1]; p2 = wilcoxon(ea(mk,HW), ea(mk,ML))[1]
                s1, s2 = f"{p1:.4f}", f"{p2:.4f}"
            else: s1 = s2 = '—'
            print(f"{et if nm=='Promedio móvil' else '':<26} "
                  f"{(f'{pv:.1%}' if nm=='Promedio móvil' else ''):>6} {nm:<16} "
                  f"{wv:>8.1%} {bv:>+8.1%} {wu:>8.1%} {bu:>+8.1%} {r2:>6.2f} {s1:>9} {s2:>9}")
            filas.append(dict(grupo=et, peso=pv, metodo=nm, wmape_val=wv, bias_val=bv,
                              wmape_u=wu, bias_u=bu, r2=r2, p_pm=s1, p_hw=s2))
        print("-"*L)

    print("\nSENSIBILIDAD — PROMEDIO MOVIL CON BLOQUES DESDE EL INICIO (no anios civiles)")
    print(f"{'GRUPO':<26} {'PM anios civiles':>18} {'PM bloques':>12} {'dif':>8}")
    for c in list(ORDEN) + ['TOTAL']:
        mk = np.ones(N,bool) if c=='TOTAL' else (lab==c)
        et = 'AGREGADO' if c=='TOTAL' else f"{NUM[c]}. {nombre[c]}"
        a = met(mk,PM)[0]; b = met(mk,PMb)[0]
        print(f"{et:<26} {a:>18.1%} {b:>12.1%} {a-b:>+8.1%}")

    d_pm = (np.abs(real-PM)*pr).sum() - (np.abs(real-ML)*pr).sum()
    d_hw = (np.abs(real-HW)*pr).sum() - (np.abs(real-ML)*pr).sum()
    print(f"\nReduccion acumulada del error valorizado frente al promedio movil: {d_pm:>+15,.0f}")
    print(f"Reduccion acumulada del error valorizado frente a Holt-Winters   : {d_hw:>+15,.0f}")

    pd.DataFrame(filas).to_csv(f'roh_{PAIS}_oe3.csv', index=False)
    pd.DataFrame({'sku': E['SKU'].values[idx], 'cluster': [NUM[c] for c in lab],
                  'nombre': [nombre[c] for c in lab]}).to_csv(f'roh_{PAIS}_clusters.csv', index=False)
    np.save(f'roh_{PAIS}_ml.npy', ML); np.save(f'roh_{PAIS}_pm.npy', PM); np.save(f'roh_{PAIS}_hw.npy', HW)
    print(f"\n[{time.time()-t0:.0f}s] Guardado: roh_{PAIS}_oe3.csv, roh_{PAIS}_clusters.csv, .npy")


# ===========================================================================
# DESPACHO
# ===========================================================================
if _CONJUNTO == 'privado':
    privado_oe1()
    correr_privado()
elif _CONJUNTO == 'm5':
    m5_oe1()
    correr_m5()
else:
    _M = 'cz' if _CONJUNTO == 'rohlik_cz' else 'hu'
    rohlik_oe1(_M)
    rohlik_oe2_oe3(_M)
