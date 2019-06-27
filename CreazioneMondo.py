import numpy as np

class mondo():
    'Costruzione Mondo'
    #così in futuro si può passare come paramentro della classe
    #per avere posizione iniziale diverso, ma andra gestita eccezione posizione
    # non valido in caso si sovrappone su X
    # per il momento il mondo è un 5x5
    def __init__(self):
        self.archer_pos = 4,1
        self.knight_pos = 3,0
        self.dragon_pos = 0,4
        self.mondo = np.matrix([['O','O','O','O','O'],
                                ['O','X','O','O','O'],
                                ['O','O','X','O','O'],
                                ['O','O','O','O','O'],
                                ['O','O','O','X','O']])
        self.mondo[self.archer_pos] = 'A'
        self.mondo[self.knight_pos] = 'K'
        self.mondo[self.dragon_pos] = 'D'
        self.end = False
        
    def displayWorld(self):
        print(self.mondo)

    def moveArcher(self,key):
        'Muove arciere'
        char = 'A'
        if self.end:
            print('Guarda che la partita è finita')
        if (key=='UP'):
            if(self.archer_pos[0]==0):
                print('non è possibile muoverlo sopra, limite mondo')
                #solleva eccezione
                return False
            elif(self.mondo[self.archer_pos[0]-1,self.archer_pos[1]]=='X'):
                print('impossibile muoversi in questa direzione, \
                       vi è un ostacolo')
                #solleva eccezione
                return False
            elif(self.mondo[self.archer_pos[0]-1,self.archer_pos[1]]=='K'):
                print('Il cavaliere ha occupato già il posto occupato')
                return False
            elif(self.mondo[self.archer_pos[0]-1,self.archer_pos[1]]=='D'):
                print("Hai ucciso il drago")
                self.mondo[self.archer_pos]='O'
                self.archer_pos = self.archer_pos[0]-1,self.archer_pos[1]  #cambia le coordinate
                self.mondo[self.archer_pos]=char
                self.end=True
                return True
            else:
                self.mondo[self.archer_pos]='O'
                self.archer_pos = self.archer_pos[0]-1,self.archer_pos[1]  #cambia le coordinate
                self.mondo[self.archer_pos]=char #mossa valida normale
                return True
        if (key=='DOWN'):
            if(self.archer_pos[0]==4):
                print('non è possibile muoverlo sopra, limite mondo')
                #solleva eccezione
                return False
            elif(self.mondo[self.archer_pos[0]+1,self.archer_pos[1]]=='X'):
                print('impossibile muoversi in questa direzione, \
                       vi è un ostacolo')
                #solleva eccezione
                return False
            elif(self.mondo[self.archer_pos[0]+1,self.archer_pos[1]]=='K'):
                print('Il cavaliere ha occupato già il posto occupato')
                return False
            elif(self.mondo[self.archer_pos[0]+1,self.archer_pos[1]]=='D'):
                print("Hai ucciso il drago")
                self.mondo[self.archer_pos]='O'
                self.archer_pos = self.archer_pos[0]+1,self.archer_pos[1]  #cambia le coordinate
                self.mondo[self.archer_pos]=char
                self.end=True
                return True
            else:
                self.mondo[self.archer_pos]='O'
                self.archer_pos = self.archer_pos[0]+1,self.archer_pos[1]  #cambia le coordinate
                self.mondo[self.archer_pos]=char #mossa valida normale
                return True
        if (key=='LEFT'):
            if(self.archer_pos[1]==0):
                print('non è possibile muoverlo sopra, limite mondo')
                #solleva eccezione
                return False
            elif(self.mondo[self.archer_pos[0],self.archer_pos[1]-1]=='X'):
                print('impossibile muoversi in questa direzione, \
                       vi è un ostacolo')
                #solleva eccezione
                return False
            elif(self.mondo[self.archer_pos[0],self.archer_pos[1]-1]=='K'):
                print('Il cavaliere ha occupato già il posto occupato')
                return False
            elif(self.mondo[self.archer_pos[0],self.archer_pos[1]-1]=='D'):
                print("Hai ucciso il drago")
                self.mondo[self.archer_pos]='O'
                self.archer_pos = self.archer_pos[0],self.archer_pos[1]-1  #cambia le coordinate
                self.mondo[self.archer_pos]=char
                self.end=True
                return True
            else:
                self.mondo[self.archer_pos]='O'
                self.archer_pos = self.archer_pos[0],self.archer_pos[1]-1  #cambia le coordinate
                self.mondo[self.archer_pos]=char #mossa valida normale
                return True
        if(key=='RIGHT'):
            if(self.archer_pos[1]==4):
                print('non è possibile muoverlo sopra, limite mondo')
                #solleva eccezione
                return False
            elif(self.mondo[self.archer_pos[0],self.archer_pos[1]+1]=='X'):
                print('impossibile muoversi in questa direzione, \
                       vi è un ostacolo')
                #solleva eccezione
                return False
            elif(self.mondo[self.archer_pos[0],self.archer_pos[1]+1]=='K'):
                print('Il cavaliere ha occupato già il posto occupato')
                return False
            elif(self.mondo[self.archer_pos[0],self.archer_pos[1]+1]=='D'):
                print("Hai ucciso il drago")
                self.mondo[self.archer_pos]='O'
                self.archer_pos = self.archer_pos[0],self.archer_pos[1]+1  #cambia le coordinate
                self.mondo[self.archer_pos]=char
                self.end=True
                return True
            else:
                self.mondo[self.archer_pos]='O'
                self.archer_pos = self.archer_pos[0],self.archer_pos[1]+1  #cambia le coordinate
                self.mondo[self.archer_pos]=char #mossa valida normale
                return True