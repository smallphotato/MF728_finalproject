class yieldCurvePredictor:

    def __init__(self, model, sim_date, data):
        self.model = model
        self.sim_date = sim_date
        self.data = data

    def actual(self):
        import pandas as pd
        return pd.DataFrame(self.data.loc[self.sim_date]).transpose()

    def simulate_curve(self):
        import pandas as pd

        actual = self.actual().transpose()
        
        simulated_rates = {}
        for column in actual.index:
            if 'DGS' in column:
                initial_rate = self.data[column].loc[self.data.index < self.sim_date].iloc[-1]
                simulated_rates[column] = self.model.next_rate(initial_rate, 1/252)

        return pd.DataFrame(simulated_rates, index=[self.sim_date])

    def simulate_bond_prices(self, face_value):
        import pandas as pd
        import numpy as np
        
        simulated_bond_prices = {}
        
        for column in self.data.columns:
            if 'DGS' in column:
                simulated_rate = self.simulate_curve()[column]/100  
                    
                maturity = (int(column.replace('DGS', '').replace('MO', '')) / 12) if 'MO' in column else int(column.replace('DGS', ''))
                # Converts nominal rate to continuous
                continuous_rate = np.log(1 + simulated_rate)
                    
                bond_price = self.model.exact_solution(continuous_rate, maturity) * face_value
                simulated_bond_prices[column] = bond_price

        return pd.DataFrame(simulated_bond_prices)

    def actual_bond_prices(self, face_value):
        import pandas as pd
        import numpy as np

        actual = self.actual()        
        actual_bond_prices = {}
        
        for column in self.data.columns:
            if 'DGS' in column:
                maturity = int(column.replace('DGS', '').replace('MO', '')) / 12 if 'MO' in column else int(column.replace('DGS', ''))
                actual_rate = actual[column]/100
                continuous_rate = np.log(1 + actual_rate)
                bond_price = self.model.exact_solution(continuous_rate, maturity) * face_value
                actual_bond_prices[column] = bond_price
        return pd.DataFrame(actual_bond_prices)

    def comb_data(self):
        import pandas as pd

        actual_rate = self.actual()
        sim_rate = self.simulate_curve()
        comb_rates =  pd.DataFrame({'Actual Rates': actual_rate.iloc[0],
                                    'Simulated Rates': sim_rate.iloc[0]})
    
        actual_price = self.actual_bond_prices(1000)
        sim_price = self.simulate_bond_prices(1000)
    
        comb_prices =  pd.DataFrame({'Actual Prices': actual_price.iloc[0],
                                     'Simulated Prices': sim_price.iloc[0]})
            
        return comb_rates, comb_prices