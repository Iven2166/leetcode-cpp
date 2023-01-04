//
//  sword_solution.h
//  leetcode
//
//  Created by  on 2023/1/5.
//

#ifndef sword_solution_h
#define sword_solution_h




class sword_solution {
public:
    // 剑指 Offer 16.数值的整数次方
    double Power(double base, int exponent){
        // 非法输入
        if(base < std::numeric_limits<double>::epsilon() && exponent < 0){
            return 0.0;
        }
        unsigned int absExponent = (unsigned int) exponent;
        if(exponent < 0)
            absExponent = (unsigned int) -exponent;
        double res = PowerCore(base, absExponent);
        if(exponent < 0)
            res = 1.0 / res;
        return res;
    }
    double PowerCore(double base, unsigned int absExponent){
        // 普通解法
//        double res = 1.0;
//        for(int i = 0; i < absExponent; i++){
//            res *= base;
//        }
//        return res;
        // 优化解法： a^n = a^(n/2) * a^(n/2); a^n = a^(n/2) * a^(n/2) * a;
        if(absExponent == 0){
            return 1.0;
        }
        if(absExponent == 1){
            return base;
        }
        double mid = PowerCore(base, absExponent >> 1);
        mid = mid * mid;
        if(absExponent & 0x1 == 1)
            mid *= base;
        return mid;        
    }
    
    int hammingWeight(uint32_t n){
        int ans = 0;
        while(n){
            ans += (n & 1);
            n >>= 1;
        }
        
        return ans;
    }
    
    int cuttingRope(int n){
        vector<int> dp(n+1);
        for(int i=2;i<n;i++){
            int curMax = 0;
            for(int j=1; j<i; j++){
                curMax = max(curMax, max(j*(i-j), j*dp[i-j]));
            }
            dp[i] = curMax;
        }
        return dp[n];
    }
};


#endif /* sword_solution_h */
