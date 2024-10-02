%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       Testing the FIS Design
%                       Using Custom Membership
%                            Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function out = customMF1(x,params)

    for i = 1:length(x)
        if x(i) == params
            y(i) = 1;
        else
            y(i) = 0;
        end
    end
    out = y;
end



