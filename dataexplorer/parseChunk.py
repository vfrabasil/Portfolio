import pandas as pd   
import time


pd.options.mode.chained_assignment = None  # default='warn'
   
chunkSize = 200000
chunk_list  = []
chunk_list2 = []
chunk_list3 = []
chunk_list4 = []
chunk_list5 = []

cont = 0
grouping_columns  = ['terminal', 'tran-cde']
grouping_columns2 = ['card-fiid', 'tran-cde']
grouping_columns3 = ['card-fiid', 'tran-cde', 'resp-cde']
#grouping_columns4 = ['term-fiid', 'tran-cde', 'resp-cde']
grouping_columns4 = ['terminal', 'tran-cde', 'resp-cde']
grouping_columns5 = ['terminal', 'card-fiid', 'tran-cde', 'resp-cde']

columns_to_show   = ['aprobadas', 'rechazadas']

#input = 'produccion2\Todas_210607_ente_10y15.csv'
#outputTerm = 'produccion2\salida_210607t.csv'
#outputCard = 'produccion2\salida_210607c.csv'
#outputRech = 'produccion2\salida_210607r.csv'

input = 'produccion5\prod_PRE.csv'
#outputTerm = f"{mys[:10]}_{mys[10:]}"

index = input.find('.csv')
outputTerm = input[:index] + '_term' + input[index:]
outputCard = input[:index] + '_card' + input[index:]
outputRespT = input[:index] + '_term' + 'Resp' + input[index:]
outputRespC = input[:index] + '_card' + 'Resp' + input[index:]
outputRespTC = input[:index] + '_PREP' + input[index:]



#outputTerm = 'produccion3\salida_t.csv'
#outputCard = 'produccion3\salida_c.csv'
#outputResp = 'produccion3\salida_r.csv'


t0= time.time()
filas = 0
#for chunk in pd.read_csv('Caso trx71_pocho_210531.csv', sep='\t', low_memory=False, dtype={'TERM_FIID': str, 'PAN': str, 'TRAN_DAT': str, 'TRAN_TIM': str, 'TERM_TYPE': str, 'RESP_CDE': str}, chunksize=chunkSize):

# 1 -- Por SQL:
for chunk in pd.read_csv(input, sep='\t', low_memory=False, dtype={'FIID_TERM': str, 'PAN': str, 'TRAN_DAT': str, 'TRAN_TIM': str, 'TERM_TYPE': str, 'RESP_CDE': str}, chunksize=chunkSize):
# 2 -- Por RCARD:
#for chunk in pd.read_csv(input, sep=';', low_memory=False, dtype={'term-fiid': str, 'pan': str, 'term-typ': str, 'resp-cde': str, 'card-fiid': str,}, chunksize=chunkSize):


    filas = filas + chunk.shape[0]
    #Tomo solo las columnas que necesito, dropeo el resto y cambio el nombre:

    # 1 -- Por SQL:
    chunk = chunk.filter(['FIID_TERM', 'FIID_CARD', 'TERM_TYPE', 'TRAN_CDE', 'RESP_CDE'])
    chunk.columns = ['term-fiid', 'card-fiid','term-typ', 'tran-cde', 'resp-cde']

    # 2 -- Por RCARD:
    # term-fiid;term-typ;card-fiid;pan;tran-cde;tipo-tran;responder;resp-cde;reversal;date;time;amt
    #chunk = chunk.filter(['term-fiid', 'card-fiid', 'term-typ', 'tran-cde', 'resp-cde'])
    #chunk.columns = ['term-fiid', 'card-fiid','term-typ', 'tran-cde', 'resp-cde']

    #filtro 1: por fiid terminal
    chunk_filter = chunk[['term-fiid','term-typ', 'tran-cde', 'resp-cde']]
    #filtro 2: por fiid tarjeta
    chunk_filter2 = chunk[['card-fiid', 'tran-cde', 'resp-cde']]
    #filtro 3: resp-cde por tran-cde (card-fiid)
    chunk_filter3 = chunk[['card-fiid', 'tran-cde', 'resp-cde']]
    #filtro 4: resp-cde por tran-cde (term-fiid)
    chunk_filter4 = chunk[['term-fiid', 'tran-cde', 'resp-cde']]
    #filtro 5: resp-cde por tran-cde (term-fiid + card-fiid)
    chunk_filter5 = chunk[['term-fiid', 'card-fiid', 'tran-cde', 'resp-cde']]


    chunk_filter['terminal'] = chunk['term-fiid'].str.strip() + "-" + chunk['term-typ'].str.strip() 
    chunk_filter['aprobadas']  = chunk["resp-cde"].apply(lambda x: 1 if (x == '000' or x == '001') else 0)
    chunk_filter['rechazadas']  = chunk["resp-cde"].apply(lambda x: 1 if (x != '000' and x != '001') else 0)
    chunk_filter.drop(['term-fiid', 'term-typ', 'resp-cde'], axis=1, inplace=True)
    
    chunk_filter2['card-fiid'] = chunk['card-fiid']
    chunk_filter2['aprobadas']  = chunk["resp-cde"].apply(lambda x: 1 if (x == '000' or x == '001') else 0)
    chunk_filter2['rechazadas']  = chunk["resp-cde"].apply(lambda x: 1 if (x != '000' and x != '001') else 0)
    chunk_filter2.drop(['resp-cde'], axis=1, inplace=True)


    #chunk_filter3['tran-cde'] = chunk['tran-cde']
    #dic = df.groupby(by='tran-cde')['tipo-tran'].unique().apply(lambda x:x.tolist()).to_dict()

    #chunk_filter3['Subtotal'] = chunk_filter3.groupby('Client')['USD_Balance'].transform('sum')
    chunk_filter3['card-fiid'] = chunk['card-fiid']
    chunk_filter3['tran-cde'] = chunk['tran-cde']
    chunk_filter3['resp-cde'] = chunk['resp-cde']
    chunk_filter3['count']  = 1

    chunk_filter4['terminal'] = chunk['term-fiid'].str.strip() + "-" + chunk['term-typ'].str.strip() 
    #chunk_filter4['term-fiid'] = chunk['term-fiid']
    chunk_filter4['tran-cde'] = chunk['tran-cde']
    chunk_filter4['resp-cde'] = chunk['resp-cde']
    chunk_filter4['count']  = 1



    # Once the data filtering is done, append the chunk to list ok
    #chunk_list.append(chunk_filter)

    #chunk_list.append(chunk_filter)
    #df_concatTerm = pd.concat(chunk_list)
    #chunk_list2.append(chunk_filter2)
    #df_concatCard = pd.concat(chunk_list2)

    chunk_list3.append(chunk_filter3)
    df_concatRespC = pd.concat(chunk_list3)

    chunk_list4.append(chunk_filter4)
    df_concatRespT = pd.concat(chunk_list4)

    chunk_list5.append(chunk_filter3)
    df_concatRespTC = pd.concat(chunk_list3)
    chunk_list5.append(chunk_filter4)
    df_concatRespTC = pd.concat(chunk_list4)

    print(f'chunk: {cont}')
    cont = cont + 1
    t1 = time.time()
    print(f'process time: {t1 - t0}')
    

    
# concat the list into dataframe ok
#df_concat = pd.concat(chunk_list)

#grouping_columns = ['terminal', 'tran-cde']
#columns_to_show = ['apr', 'rec']
#df_concat = df_concat.groupby(by=grouping_columns)[columns_to_show].sum().reset_index()

#df_concat['ratio']  = df_concat["resp-cde"].apply(lambda x: 1 if (x != '000' and x != '001') else 0)

#print('\n \n * Concatenado antes del group by: ')
#df_concat.info()
#df_concatTerm   = df_concatTerm.groupby(by=grouping_columns)[columns_to_show].sum().reset_index()
#df_concatCard   = df_concatCard.groupby(by=grouping_columns2)[columns_to_show].sum().reset_index()
df_concatRespC  = df_concatRespC.groupby(by=grouping_columns3)['count'].sum().reset_index()
df_concatRespT  = df_concatRespT.groupby(by=grouping_columns4)['count'].sum().reset_index()
df_concatRespTC = df_concatRespT.append(df_concatRespC)
#print('\n \n * Concatenado despues del group by: ')
#df_concat.info()

#df_concatRespTC['card-fiid'] = df_concatRespTC['card-fiid'].apply(lambda x: '{0:0>4}'.format(x))


print('\n \n')
print("Resultado:")

#print("**********")
#print(df_concatTerm)
#df_concatTerm.to_csv(outputTerm, sep=';', index=False)
#print("**********")
#print(df_concatCard)
#df_concatCard.to_csv(outputCard, sep=';', index=False)
#print("**********")
#print(df_concatRespC)
#df_concatRespC.to_csv(outputRespC, sep=';', index=False)
#print("**********")
#print(df_concatRespT)
#df_concatRespT.to_csv(outputRespT, sep=';', index=False)
print("**********")
print(df_concatRespTC)
df_concatRespTC.to_csv(outputRespTC, sep=';', index=False)
print("**********")

t2 = time.time()
print(f'total time: {t2 - t0}')
print(f'total registros: {filas}')
print('\n \n')