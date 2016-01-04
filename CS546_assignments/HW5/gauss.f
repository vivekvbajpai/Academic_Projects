!      Gauss Elimination without pivoting 

program gauss
       integer n,row,col,norm
       parameter (n=256)
       real  X(n),B(n),A(n,n),multiplier
       real*8 elapsed1, elapsed2,rtc,elapsedp1,elapsedp2
      
	elapsed1 = rtc()

! --------- Initialise all elements to Random Values.
       do row = 1,n 
         do col = 1,n 
             A(row,col) = (1.0 * irand())/32768.0
         enddo
         B(row) = (1.0 * irand())/32768.0
       enddo


! -------------------------------------------------
! Parallelized part
       elapsedp1 = rtc()
 	   
!HPF$ PROCESSORS PR(1,4)

!Align B with A as each element in B correspond to each row in A
!HPF$ ALIGN B(:) with A(:,*)

!distribute the array A and B using cyclic distribution
!HPF$ DISTRIBUTE A(CYCLIC,*)
      	  
	   do norm = 1 , n-1
		 forall(row=(norm:n))						!for all row present in A after norm, lets normalize the matrix 
			multiplier=A(row,norm)/A(norm,norm)		! calculating the multiplier for each iteration
			A(row,:) = A(row,:) - (A(norm,:)* multiplier) !Normalizing the entire row at once. Thi type of opertion is allowed in HPF
			B(row) = B(row) - B(norm) * multiplier	!update B accordingly
       enddo
	elapsedp2 = rtc()

! -------------------------------- backsubstitute
        do row=n-1,1,-1
            X(row) = B(row)
            do col = n-1,row+1,-1  
               X(row) = X(row) - A(row,col) * X(col)
	    enddo
            X(row) = X(row)/ A(row,row)
        enddo
	elapsed2 = rtc()

! -----------------------Check correctness of code
	do row=1,n
	  do col=1,row -1
	     if ( A(row,col) .GT. 1e-3 ) print *,"Error in",row,col,A(row,col)
	  enddo
	enddo
	print *,"Elapsed Time", elapsed2 - elapsed1
	print *,"Elapsed Time in elimination phase", elapsedp2 - elapsedp1
      stop
end
