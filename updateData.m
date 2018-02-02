function CRFDATA = updateData(CRFDATA, vars, COUNT_UNARY, COUNT_FACTORS)

   CRFDATA.vars = vars;
   if nargin >= 3 & ~isempty(COUNT_UNARY)
      CRFDATA.COUNT_UNARY = COUNT_UNARY;
   end;
   if nargin >= 4 & ~isempty(COUNT_FACTORS)
      CRFDATA.COUNT_FACTORS = COUNT_FACTORS;
   end;